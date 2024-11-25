import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename
import xml.etree.ElementTree as ET
import pandas as pd
from joblib import load
import numpy as np
from svgpathtools import parse_path, Line, CubicBezier, QuadraticBezier
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from PyPDF2 import PdfMerger
import io

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'processed'
MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'svg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Límite de 16MB

# Crear las carpetas necesarias
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Cargar modelos y escaladores al inicio
rect_model = load('rect_classifier_model.joblib')
rect_scaler = load('rect_classifier_scaler.joblib')
path_model = load('modelo_random_forest_paths.pkl')
text_classifier = load( 'svg_classifier_textos.joblib')
text_scaler = load( 'scaler_textos.joblib')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_rect_features(rect):
    x = float(rect.get('x', '0'))
    y = float(rect.get('y', '0'))
    width = float(rect.get('width', '0'))
    height = float(rect.get('height', '0'))
    hierarchy_level = len(rect.findall(".."))
    bounding_box_area = width * height
    aspect_ratio = width / height if height != 0 else 0
    return [x, y, width, height, hierarchy_level, bounding_box_area, aspect_ratio]

def extract_path_features(d):
    path = parse_path(d)
    d_length = path.length()
    num_commands = len(path)
    num_coordinates = sum(len(seg) for seg in path)
    avg_coordinate = np.mean([coord.real for seg in path for coord in seg])
    return d_length, num_commands, num_coordinates, avg_coordinate

def extract_enhanced_path_features(d):
    path = parse_path(d)
    d_length = path.length()
    num_commands = len(path)
    
    command_types = [seg.__class__.__name__ for seg in path]
    lines = sum(1 for cmd in command_types if cmd == 'Line')
    cubic_beziers = sum(1 for cmd in command_types if cmd == 'CubicBezier')
    quadratic_beziers = sum(1 for cmd in command_types if cmd == 'QuadraticBezier')
    
    line_ratio = lines / num_commands if num_commands > 0 else 0
    bezier_ratio = (cubic_beziers + quadratic_beziers) / num_commands if num_commands > 0 else 0
    
    x_coords = [coord.real for seg in path for coord in seg]
    y_coords = [coord.imag for seg in path for coord in seg]
    
    if x_coords and y_coords:
        x_std = np.std(x_coords)
        y_std = np.std(y_coords)
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        bbox_width = x_range
        bbox_height = y_range
        bbox_area = bbox_width * bbox_height
        aspect_ratio = bbox_width / (bbox_height + 1e-10)
        
        height_variance = np.var([seg.length() for seg in path if isinstance(seg, Line)])
        height_consistency = height_variance / (bbox_height ** 2 + 1e-10)
        
        horizontal_threshold = bbox_height * 0.1
        horizontal_segments = sum(1 for i in range(len(y_coords)-1)
                                if abs(y_coords[i] - y_coords[i+1]) < horizontal_threshold)
        horizontal_ratio = horizontal_segments / (len(y_coords) - 1) if len(y_coords) > 1 else 0
        
        curve_angles = []
        for seg in path:
            if isinstance(seg, CubicBezier):
                control1, control2 = seg.control1, seg.control2
                start, end = seg.start, seg.end
                
                v1 = control1 - start
                v2 = control2 - control1
                v3 = end - control2
                
                angles = []
                if abs(v1) > 1e-10 and abs(v2) > 1e-10:
                    angles.append(abs(np.angle(v2 / v1)))
                if abs(v2) > 1e-10 and abs(v3) > 1e-10:
                    angles.append(abs(np.angle(v3 / v2)))
                
                curve_angles.extend(angles)
                
            elif isinstance(seg, QuadraticBezier):
                control = seg.control
                start, end = seg.start, seg.end
                
                v1 = control - start
                v2 = end - control
                
                if abs(v1) > 1e-10 and abs(v2) > 1e-10:
                    curve_angles.append(abs(np.angle(v2 / v1)))
        
        curve_angle_variance = np.var(curve_angles) if curve_angles else 0
        
        segment_lengths = [seg.length() for seg in path]
        length_variance = np.var(segment_lengths) if segment_lengths else 0
        length_regularity = length_variance / (np.mean(segment_lengths) ** 2 + 1e-10)
        
        path_density = d_length / (bbox_area + 1e-10)
        points_density = len(x_coords) / (bbox_area + 1e-10)
        
        left_half = sum(1 for x in x_coords if x < (min(x_coords) + max(x_coords))/2)
        right_half = len(x_coords) - left_half
        symmetry_ratio = min(left_half, right_half) / max(left_half, right_half) if max(left_half, right_half) > 0 else 0
        
    else:
        x_std = y_std = bbox_width = bbox_height = bbox_area = aspect_ratio = 0
        height_consistency = horizontal_ratio = curve_angle_variance = 0
        length_regularity = path_density = points_density = symmetry_ratio = 0
    
    return [
        d_length, num_commands, line_ratio, bezier_ratio,
        x_std, y_std, bbox_width, bbox_height, bbox_area,
        aspect_ratio, height_consistency, horizontal_ratio,
        curve_angle_variance, length_regularity, path_density,
        points_density, symmetry_ratio
    ]

def process_rects(input_file, output_file):
    tree = ET.parse(input_file)
    root = tree.getroot()
    svg_attributes = root.attrib
    
    codigos_layer = root.find(".//*[@id='CODIGOS']")
    if codigos_layer is None:
        codigos_layer = ET.Element('g', id='CODIGOS')
    else:
        codigos_layer.clear()
    
    rects_and_parents = [(rect, parent) for parent in root.iter() 
                         for rect in parent.findall("{http://www.w3.org/2000/svg}rect")]
    
    features = [extract_rect_features(rect) for rect, _ in rects_and_parents]
    X = pd.DataFrame(features, columns=['x', 'y', 'width', 'height', 
                                      'hierarchy_level', 'bounding_box_area', 'aspect_ratio'])
    X_scaled = rect_scaler.transform(X)
    predictions = rect_model.predict(X_scaled)
    
    for (rect, parent), prediction in zip(rects_and_parents, predictions):
        if prediction == 1:
            parent.remove(rect)
            codigos_layer.append(rect)
        else:
            parent.remove(rect)
    
    new_root = ET.Element('svg', svg_attributes)
    new_root.append(codigos_layer)
    new_tree = ET.ElementTree(new_root)
    new_tree.write(output_file, encoding='utf-8', xml_declaration=True)
    return True

def process_paths(input_file, output_file):
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}
    svg_attributes = root.attrib
    
    svg_new = ET.Element('svg', attrib=svg_attributes)
    svg_new.set("xmlns", "http://www.w3.org/2000/svg")
    svg_new.set("version", "1.1")
    
    codigos_group = ET.Element('g', {'id': 'CODIGOS'})
    
    for path_elem in root.findall('.//svg:path', namespaces):
        d = path_elem.get('d')
        if d:
            features = extract_path_features(d)
            label_pred = path_model.predict([features])[0]
            if label_pred == 1:
                codigos_group.append(path_elem)
    
    svg_new.append(codigos_group)
    tree_new = ET.ElementTree(svg_new)
    tree_new.write(output_file, encoding="UTF-8", xml_declaration=True)
    return True

def process_text(input_file, output_file):
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}
    style_elem = root.find(".//svg:style", namespaces)
    defs_elem = root.find(".//svg:defs", namespaces)
    svg_attributes = root.attrib
    
    svg_new = ET.Element('svg', attrib=svg_attributes)
    svg_new.set("xmlns", "http://www.w3.org/2000/svg")
    svg_new.set("version", "1.1")
    
    if defs_elem is not None:
        svg_new.append(defs_elem)
    if style_elem is not None:
        svg_new.append(style_elem)
    
    textos_group = ET.Element('g', {'id': 'TEXTOS'})
    total_paths = texto_paths = 0
    
    for path_elem in root.findall('.//svg:path', namespaces):
        total_paths += 1
        d = path_elem.get('d')
        if d:
            try:
                features = extract_enhanced_path_features(d)
                features_scaled = text_scaler.transform([features])
                prediction = text_classifier.predict(features_scaled)[0]
                probabilities = text_classifier.predict_proba(features_scaled)[0]
                
                if prediction == 1 and probabilities[1] >= 0.6:
                    texto_paths += 1
                    path_new = ET.Element('path', path_elem.attrib)
                    path_new.set('data-confidence', f"{probabilities[1]:.2f}")
                    textos_group.append(path_new)
                    
            except Exception as e:
                continue
    
    svg_new.append(textos_group)
    tree_new = ET.ElementTree(svg_new)
    tree_new.write(output_file, encoding="UTF-8", xml_declaration=True)
    return True

def convert_svg_to_pdf(svg_path, pdf_path):
    """Convierte un archivo SVG a PDF."""
    drawing = svg2rlg(svg_path)
    renderPDF.drawToFile(drawing, pdf_path)

def merge_pdfs(pdf_paths, output_path):
    """Combina múltiples PDFs en uno solo."""
    merger = PdfMerger()
    for pdf_path in pdf_paths:
        merger.append(pdf_path)
    with open(output_path, 'wb') as output_file:
        merger.write(output_file)
    merger.close()

@app.route('/', methods=['GET', 'POST'])
def upload_svg():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No se seleccionó ningún archivo'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            base_filename = os.path.splitext(filename)[0]
            
            # Definir rutas para los archivos procesados
            output_filepath_rect = os.path.join(app.config['OUTPUT_FOLDER'], f"rect_processed_{filename}")
            output_filepath_path = os.path.join(app.config['OUTPUT_FOLDER'], f"path_processed_{filename}")
            output_filepath_text = os.path.join(app.config['OUTPUT_FOLDER'], f"text_processed_{filename}")
            
            # Definir rutas para los PDFs intermedios
            pdf_rect = os.path.join(app.config['OUTPUT_FOLDER'], f"rect_processed_{base_filename}.pdf")
            pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], f"path_processed_{base_filename}.pdf")
            pdf_text = os.path.join(app.config['OUTPUT_FOLDER'], f"text_processed_{base_filename}.pdf")
            
            # Ruta para el PDF final combinado
            combined_pdf = os.path.join(app.config['OUTPUT_FOLDER'], f"combined_processed_{base_filename}.pdf")
            
            try:
                file.save(input_filepath)
                
                # Procesar SVGs
                process_rects(input_filepath, output_filepath_rect)
                process_paths(input_filepath, output_filepath_path)
                process_text(input_filepath, output_filepath_text)
                
                # Convertir SVGs a PDFs
                convert_svg_to_pdf(output_filepath_rect, pdf_rect)
                convert_svg_to_pdf(output_filepath_path, pdf_path)
                convert_svg_to_pdf(output_filepath_text, pdf_text)
                
                # Combinar PDFs
                merge_pdfs([pdf_rect, pdf_path, pdf_text], combined_pdf)
                
                # Limpiar archivos intermedios
                for pdf_file in [pdf_rect, pdf_path, pdf_text]:
                    if os.path.exists(pdf_file):
                        os.remove(pdf_file)
                
                return jsonify({
                    'success': True,
                    'message': f'Archivo {filename} procesado exitosamente',
                    'download_url': url_for('download_file', 
                                          filename=f"combined_processed_{base_filename}.pdf")
                })
                
            except Exception as e:
                return jsonify({'error': f'Error: {str(e)}'})
        
        return jsonify({'error': 'Tipo de archivo no permitido. Solo se aceptan SVG'})
    
    return render_template('upload.html')

@app.route('/download/<filename>')
def download_file(filename):
    # Asegúrate de que el archivo exista en el directorio
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.exists(file_path):
        abort(404, description="Archivo no encontrado")

    # Enviar el archivo como attachment
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)