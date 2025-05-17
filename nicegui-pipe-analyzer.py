import os
import json
import base64
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import io
from datetime import datetime
from nicegui import ui, app
from dotenv import load_dotenv
from openai import OpenAI
import tempfile

# ----- CONFIGURATION -----
# Load environment variables from .env file
load_dotenv("../Zehnder/POC_demo/Streamlit-Chatbot/app/.env")

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables")

DEFAULT_MODEL = "gpt-4o"
OUTPUT_DIR = "pipe_extractor_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- DATA STORAGE -----
# We'll store the application state here
class AppState:
    def __init__(self):
        self.image_path = None
        self.image_base64 = None
        self.segments = []
        self.edited_segments = {}  # Track which segments were edited by humans
        self.image_width = 0
        self.image_height = 0
        self.model = DEFAULT_MODEL
        self.extraction_complete = False
        self.display_mode = "overlay"  # overlay, skeleton, or original
        self.line_width = 3
        self.highlighted_segment = None
        
state = AppState()

# ----- HELPER FUNCTIONS -----
def encode_image(image_path):
    """Encode an image file to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_pipe_segments(image_path, model=DEFAULT_MODEL):
    """Extract pipe segments from image using AI vision"""
    ui.notify("Analyzing image with AI...", type="info")
    
    try:
        # Encode image to base64
        img_b64 = encode_image(image_path)
        state.image_base64 = img_b64
        
        # Load the image to get dimensions
        img = Image.open(image_path)
        state.image_width, state.image_height = img.size
        
        # Initialize OpenAI client
        client = OpenAI(api_key=API_KEY)
        
        # Call the AI model
        ui.notify(f"Analyzing image with {model}...", type="info")
        
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a specialized vision model for extracting pipe segment data from engineering drawings and maps. You analyze images and identify highlighted pipe segments, extracting their properties and returning them only as structured JSON data without explanations."},
                {"role": "user", "content": [
                    {"type": "text", "text": 
                        "Extract all pipe segments highlighted in pink/magenta/red. "
                        "I need you to directly extract this data without explanations. "
                        "Return ONLY a JSON array of objects with fields: "
                        "id (segment identifier), "
                        "start_node (name or description of where segment begins), "
                        "end_node (name or description of where segment ends), "
                        "coords (list of [x,y] pixel coordinates for the skeleton), "
                        "pipe_type (if visible, e.g., '8\" PVC'), "
                        "length_ft (length in feet), and "
                        "confidence (0.0-1.0 indicating extraction confidence). "
                        "Do not include any explanatory text, only the JSON array."
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]}
            ]
        )
        
        ui.notify("AI analysis complete", type="positive")
        
        # Extract JSON portion from the response
        content = resp.choices[0].message.content
        
        # Try to find JSON within the response
        json_start = content.find("[")
        json_end = content.rfind("]") + 1
        
        # If no JSON found or model returned explanatory text instead
        if json_start == -1 or json_end == 0:
            ui.notify("Model didn't return proper JSON. Using fallback data...", type="warning")
            # Use fallback data for demonstration purposes
            segments = [
                {
                    "id": "MH 8-155A",
                    "start_node": "PINEY POINT DR NW",
                    "end_node": "MH 8-165",
                    "coords": [[200, 100], [220, 120], [240, 140]],
                    "pipe_type": "8\" PVC",
                    "length_ft": 226.0,
                    "confidence": 0.94
                },
                {
                    "id": "MH 8-165",
                    "start_node": "MH 8-155A",
                    "end_node": "MH 8-164",
                    "coords": [[240, 140], [260, 160], [280, 180]],
                    "pipe_type": "8\" PVC",
                    "length_ft": 210.0,
                    "confidence": 0.92
                },
                {
                    "id": "MH 8-164",
                    "start_node": "MH 8-165",
                    "end_node": "MH 8-177",
                    "coords": [[280, 180], [300, 200], [320, 220]],
                    "pipe_type": "8\" PVC",
                    "length_ft": 251.0,
                    "confidence": 0.90
                },
                {
                    "id": "MH 8-177",
                    "start_node": "MH 8-164",
                    "end_node": "MH 8-163",
                    "coords": [[320, 220], [340, 240], [360, 260]],
                    "pipe_type": "8\" PVC",
                    "length_ft": 305.0,
                    "confidence": 0.89
                },
                {
                    "id": "MH 8-163",
                    "start_node": "MH 8-177",
                    "end_node": "MH 8-145",
                    "coords": [[360, 260], [380, 280], [400, 300]],
                    "pipe_type": "8\" PVC",
                    "length_ft": 330.0,
                    "confidence": 0.86
                },
                {
                    "id": "MH 8-145",
                    "start_node": "MH 8-163",
                    "end_node": "MH 8-148",
                    "coords": [[400, 300], [420, 320], [440, 340]],
                    "pipe_type": "8\" PVC",
                    "length_ft": 211.0,
                    "confidence": 0.88
                },
                {
                    "id": "MH 8-148",
                    "start_node": "MH 8-145",
                    "end_node": "WARNKE RD NW",
                    "coords": [[440, 340], [460, 360], [480, 380]],
                    "pipe_type": "8\" PVC",
                    "length_ft": 390.0,
                    "confidence": 0.91
                }
            ]
        else:
            try:
                json_str = content[json_start:json_end]
                segments = json.loads(json_str)
                ui.notify(f"Successfully extracted {len(segments)} pipe segments", type="positive")
            except json.JSONDecodeError as e:
                ui.notify(f"JSON parsing error: {e}. Using fallback data...", type="warning")
                # Fallback data (shortened version)
                segments = [
                    {
                        "id": "MH 8-155A",
                        "start_node": "PINEY POINT DR NW",
                        "end_node": "MH 8-165",
                        "coords": [[200, 100], [220, 120], [240, 140]],
                        "pipe_type": "8\" PVC", 
                        "length_ft": 226.0,
                        "confidence": 0.94
                    },
                    {
                        "id": "MH 8-165", 
                        "start_node": "MH 8-155A",
                        "end_node": "MH 8-164",
                        "coords": [[240, 140], [260, 160], [280, 180]],
                        "pipe_type": "8\" PVC",
                        "length_ft": 210.0,
                        "confidence": 0.92
                    }
                ]
        
        # Scale coordinates if necessary to match the canvas size
        # This assumes the AI model returns coordinates based on the original image size
        state.segments = segments
        state.extraction_complete = True
        
        return segments
    except Exception as e:
        ui.notify(f"Error: {str(e)}", type="negative")
        return []

def export_to_csv():
    """Export segments to CSV file"""
    if not state.segments:
        ui.notify("No data to export", type="warning")
        return None
    
    try:
        # Create a simplified version without coords for readability
        segments_for_csv = []
        for i, seg in enumerate(state.segments):
            seg_copy = seg.copy()
            if 'coords' in seg_copy:
                del seg_copy['coords']  # Remove coords to make CSV readable
            
            # Add human edit flag
            seg_copy['human_edited'] = i in state.edited_segments
            
            segments_for_csv.append(seg_copy)
        
        # Generate CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{OUTPUT_DIR}/pipe_segments_{timestamp}.csv"
        
        df = pd.DataFrame(segments_for_csv)
        df.to_csv(output_path, index=False)
        
        ui.notify(f"Data exported to {output_path}", type="positive")
        return output_path
    except Exception as e:
        ui.notify(f"Error exporting to CSV: {e}", type="negative")
        return None

def generate_canvas_image():
    """Generate an image of the current view to display"""
    if not state.image_path or not state.segments:
        return None
    
    try:
        # Open the original image
        img = Image.open(state.image_path)
        
        # If in skeleton-only mode, create a blank canvas instead
        if state.display_mode == "skeleton":
            img = Image.new('RGBA', (state.image_width, state.image_height), (255, 255, 255, 255))
        
        # Create a drawing context
        draw = ImageDraw.Draw(img)
        
        # Draw all segments
        for i, segment in enumerate(state.segments):
            # Check if segment is highlighted
            is_highlighted = state.highlighted_segment == i
            
            # Skip if in "original" mode and not highlighted
            if state.display_mode == "original" and not is_highlighted:
                continue
            
            # Draw the pipe segment
            line_color = "yellow" if is_highlighted else "cyan"
            coords = segment.get('coords', [])
            
            # Draw the line with the specified width
            for j in range(len(coords) - 1):
                draw.line([tuple(coords[j]), tuple(coords[j+1])], 
                          fill=line_color, width=state.line_width)
            
            # Only add labels if not in original mode
            if state.display_mode != "original":
                # Draw segment label at midpoint
                if len(coords) > 0:
                    midpoint = coords[len(coords) // 2]
                    
                    # Draw start node label (green)
                    start_coord = coords[0]
                    start_node = segment.get('start_node', 'Start')
                    
                    # Background for start label
                    text_size = 12
                    text_width = len(start_node) * text_size // 2
                    draw.rectangle([
                        (start_coord[0] - text_width//2 - 4, start_coord[1] - 25),
                        (start_coord[0] + text_width//2 + 4, start_coord[1] - 5)
                    ], fill=(0, 0, 0, 180))
                    
                    # Start text
                    draw.text((start_coord[0], start_coord[1] - 15), start_node, 
                              fill='#2ecc71', anchor="mm")
                    
                    # End node label (red)
                    end_coord = coords[-1]
                    end_node = segment.get('end_node', 'End')
                    
                    # Background for end label
                    text_width = len(end_node) * text_size // 2
                    draw.rectangle([
                        (end_coord[0] - text_width//2 - 4, end_coord[1] - 25),
                        (end_coord[0] + text_width//2 + 4, end_coord[1] - 5)
                    ], fill=(0, 0, 0, 180))
                    
                    # End text
                    draw.text((end_coord[0], end_coord[1] - 15), end_node, 
                              fill='#e74c3c', anchor="mm")
                    
                    # Segment label
                    label = f"{segment.get('id', f'Segment {i+1}')}: {segment.get('length_ft', 0):.1f}ft"
                    
                    # Background for label
                    text_width = len(label) * text_size // 2
                    draw.rectangle([
                        (midpoint[0] + 5 - 4, midpoint[1] - 20 - 4),
                        (midpoint[0] + 5 + text_width + 4, midpoint[1] - 20 + text_size + 4)
                    ], fill=(0, 0, 0, 180))
                    
                    # Label text
                    draw.text((midpoint[0] + 5, midpoint[1] - 5), label, 
                              fill='cyan' if not is_highlighted else 'yellow')
        
        # Convert to bytes for display
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return buffer
    except Exception as e:
        ui.notify(f"Error generating canvas: {e}", type="negative")
        return None

def update_display():
    """Update the display canvas with current data"""
    if canvas_container.content.children:
        canvas_container.clear()
    
    if not state.image_path:
        with canvas_container:
            ui.label("No image loaded").classes('text-lg text-center')
        return
    
    # Generate the visualization
    img_bytes = generate_canvas_image()
    if img_bytes:
        with canvas_container:
            ui.image(img_bytes).classes('w-full h-auto border-2 border-gray-300')

def calculate_stats():
    """Calculate statistics about the extracted segments"""
    if not state.segments:
        return {
            "total_segments": 0,
            "total_length": 0,
            "avg_length": 0,
            "low_confidence": 0
        }
    
    total_segments = len(state.segments)
    total_length = sum(seg.get("length_ft", 0) for seg in state.segments)
    avg_length = total_length / total_segments if total_segments > 0 else 0
    low_confidence = sum(1 for seg in state.segments if seg.get("confidence", 0) < 0.7)
    
    return {
        "total_segments": total_segments,
        "total_length": round(total_length, 1),
        "avg_length": round(avg_length, 1),
        "low_confidence": low_confidence
    }

def update_summary():
    """Update the summary statistics panel"""
    stats = calculate_stats()
    
    # Update the summary items
    total_segments_label.set_text(str(stats["total_segments"]))
    total_length_label.set_text(f"{stats['total_length']} ft")
    avg_length_label.set_text(f"{stats['avg_length']} ft")
    low_confidence_label.set_text(str(stats["low_confidence"]))

def update_segment_table():
    """Update the data table with current segments"""
    # Clear existing rows
    segment_table.clear()
    
    # Add headers
    with segment_table:
        with ui.row().classes('w-full font-bold bg-blue-100 p-2'):
            ui.label('ID').classes('w-1/7')
            ui.label('Start Node').classes('w-1/7')
            ui.label('End Node').classes('w-1/7')
            ui.label('Pipe Type').classes('w-1/7')
            ui.label('Length (ft)').classes('w-1/7')
            ui.label('Confidence').classes('w-1/7')
            ui.label('Actions').classes('w-1/7')
    
    # Add segment rows
    for i, segment in enumerate(state.segments):
        with segment_table:
            # Highlight row if it's the selected segment
            row_classes = 'w-full p-2'
            if state.highlighted_segment == i:
                row_classes += ' bg-yellow-100'
            elif i in state.edited_segments:
                row_classes += ' bg-green-100'  # Mark human-edited rows
            elif segment.get('confidence', 0) < 0.7:
                row_classes += ' bg-red-100'  # Mark low confidence rows
                
            with ui.row().classes(row_classes):
                ui.label(segment.get('id', f'Segment {i+1}')).classes('w-1/7')
                ui.label(segment.get('start_node', 'Unknown')).classes('w-1/7')
                ui.label(segment.get('end_node', 'Unknown')).classes('w-1/7')
                ui.label(segment.get('pipe_type', 'Unknown')).classes('w-1/7')
                ui.label(f"{segment.get('length_ft', 0):.1f}").classes('w-1/7')
                
                # Confidence with color
                conf = segment.get('confidence', 0)
                conf_text = f"{conf*100:.0f}%"
                conf_color = "text-red-500" if conf < 0.7 else "text-orange-500" if conf < 0.9 else "text-green-500"
                ui.label(conf_text).classes(f'w-1/7 {conf_color}')
                
                # Actions
                with ui.element().classes('w-1/7 flex gap-2'):
                    ui.button('Highlight', icon='search', on_click=lambda i=i: set_highlight(i)) \
                        .classes('px-2 py-1 text-xs')
                    ui.button('Edit', icon='edit', on_click=lambda i=i: open_edit_dialog(i)) \
                        .classes('px-2 py-1 text-xs')

def set_highlight(index):
    """Set the highlighted segment"""
    state.highlighted_segment = index
    # Update UI
    update_display()
    update_segment_table()

def open_edit_dialog(index):
    """Open dialog to edit a segment"""
    segment = state.segments[index]
    
    with ui.dialog() as dialog, ui.card().classes('w-96'):
        ui.label(f"Edit Segment {segment.get('id', f'#{index+1}')}").classes('text-xl font-bold mb-4')
        
        id_input = ui.input('ID', value=segment.get('id', f'Segment {index+1}')).classes('w-full mb-2')
        start_input = ui.input('Start Node', value=segment.get('start_node', 'Unknown')).classes('w-full mb-2')
        end_input = ui.input('End Node', value=segment.get('end_node', 'Unknown')).classes('w-full mb-2')
        pipe_input = ui.input('Pipe Type', value=segment.get('pipe_type', 'Unknown')).classes('w-full mb-2')
        length_input = ui.number('Length (ft)', value=segment.get('length_ft', 0), format='%.1f').classes('w-full mb-2')
        confidence_input = ui.slider('Confidence', min=0, max=1, step=0.01, value=segment.get('confidence', 0.8)).classes('w-full mb-4')
        
        with ui.row().classes('justify-end gap-2 mt-4'):
            ui.button('Cancel', on_click=dialog.close).classes('bg-gray-300')
            
            def save_changes():
                # Update the segment
                state.segments[index]['id'] = id_input.value
                state.segments[index]['start_node'] = start_input.value
                state.segments[index]['end_node'] = end_input.value
                state.segments[index]['pipe_type'] = pipe_input.value
                state.segments[index]['length_ft'] = float(length_input.value)
                state.segments[index]['confidence'] = confidence_input.value
                
                # Mark as human-edited
                state.edited_segments[index] = True
                
                # Update UI
                update_display()
                update_segment_table()
                update_summary()
                dialog.close()
                ui.notify(f"Segment {id_input.value} updated", type="positive")
                
            ui.button('Save', on_click=save_changes).classes('bg-blue-500 text-white')
        
        dialog.open()

# ----- UI LAYOUT -----
# Create the main UI layout
@ui.page('/')
def main_page():
    ui.add_head_html('''
    <style>
        .summary-item {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        .summary-title {
            font-size: 0.9rem;
            color: #7f8c8d;
            margin-bottom: 0.25rem;
        }
        .summary-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
        }
    </style>
    ''')
    
    # Header
    with ui.header().classes('bg-blue-800 text-white p-4'):
        ui.label('Pipe Segment Analyzer').classes('text-xl font-bold')
    
    # Main content
    with ui.column().classes('p-4 gap-4 w-full max-w-screen-xl mx-auto'):
        # File upload and model selection
        with ui.card().classes('w-full'):
            ui.label('Load Image & Configure').classes('text-lg font-bold mb-2')
            
            with ui.row().classes('items-center gap-4'):
                # Image upload
                upload = ui.upload(
                    label='Upload pipe map or drawing',
                    multiple=False,
                    on_upload=lambda e: handle_upload(e)
                ).props('accept=".jpg,.jpeg,.png"').classes('w-1/3')
                
                # Model selection
                model_select = ui.select(
                    ['gpt-4o', 'gpt-4-vision-preview'], 
                    label='AI Model', 
                    value=DEFAULT_MODEL,
                    on_change=lambda e: set_model(e)
                ).classes('w-1/3')
                
                # Process button
                ui.button('Process Image', on_click=process_image).classes('w-1/3 bg-blue-500 text-white')
        
        # Summary statistics
        with ui.card().classes('w-full') as summary_card:
            ui.label('Summary').classes('text-lg font-bold mb-2')
            
            with ui.row().classes('gap-4'):
                # Summary items will be updated dynamically
                with ui.column().classes('summary-item w-1/4'):
                    ui.label('Total Segments').classes('summary-title')
                    global total_segments_label
                    total_segments_label = ui.label('0').classes('summary-value')
                
                with ui.column().classes('summary-item w-1/4'):
                    ui.label('Total Length').classes('summary-title')
                    global total_length_label
                    total_length_label = ui.label('0 ft').classes('summary-value')
                
                with ui.column().classes('summary-item w-1/4'):
                    ui.label('Avg. Segment Length').classes('summary-title')
                    global avg_length_label
                    avg_length_label = ui.label('0 ft').classes('summary-value')
                
                with ui.column().classes('summary-item w-1/4'):
                    ui.label('Low Confidence Segments').classes('summary-title')
                    global low_confidence_label
                    low_confidence_label = ui.label('0').classes('summary-value')
        
        # Visualization controls
        with ui.card().classes('w-full'):
            ui.label('Display Settings').classes('text-lg font-bold mb-2')
            
            with ui.row().classes('items-center gap-4'):
                # View mode - NiceGUI requires the value to be one of the exact values in the options
                view_mode = ui.select(
                    options=['overlay', 'skeleton', 'original'],
                    label='View Mode',
                    value='overlay',
                    on_change=lambda e: set_view_mode(e.value)
                ).classes('w-1/3')
                
                # Line width
                line_width = ui.slider(
                    min=1, max=10, step=1, value=state.line_width,
                    label='Line Width',
                    on_change=lambda e: set_line_width(e.value)
                ).classes('w-1/3')
                
                # Export buttons
                with ui.element().classes('w-1/3 flex gap-2'):
                    ui.button('Export CSV', icon='download', on_click=export_to_csv).classes('bg-green-500 text-white')
                    ui.button('Export Image', icon='image', on_click=export_image).classes('bg-green-500 text-white')
        
        # Canvas for visualization
        with ui.card().classes('w-full h-96 overflow-auto') as canvas_card:
            ui.label('Visualization').classes('text-lg font-bold mb-2')
            
            global canvas_container
            canvas_container = ui.element().classes('w-full h-full relative')
            with canvas_container:
                ui.label('No image loaded').classes('text-lg text-center')
        
        # Data table
        with ui.card().classes('w-full') as table_card:
            ui.label('Segment Data').classes('text-lg font-bold mb-2')
            
            global segment_table
            segment_table = ui.element().classes('w-full border-collapse')
            with segment_table:
                with ui.row().classes('w-full font-bold bg-blue-100 p-2'):
                    ui.label('ID').classes('w-1/7')
                    ui.label('Start Node').classes('w-1/7')
                    ui.label('End Node').classes('w-1/7')
                    ui.label('Pipe Type').classes('w-1/7')
                    ui.label('Length (ft)').classes('w-1/7')
                    ui.label('Confidence').classes('w-1/7')
                    ui.label('Actions').classes('w-1/7')
        
        # Footer with legend
        with ui.card().classes('w-full'):
            ui.label('Legend').classes('text-lg font-bold mb-2')
            
            with ui.row().classes('gap-4 items-center'):
                with ui.element().classes('flex items-center gap-2'):
                    ui.element().classes('w-4 h-4 bg-cyan-500')
                    ui.label('Pipe Segment')
                
                with ui.element().classes('flex items-center gap-2'):
                    ui.element().classes('w-4 h-4 bg-yellow-500')
                    ui.label('Highlighted Segment')
                
                with ui.element().classes('flex items-center gap-2'):
                    ui.element().classes('w-4 h-4 bg-green-500')
                    ui.label('Start Node')
                
                with ui.element().classes('flex items-center gap-2'):
                    ui.element().classes('w-4 h-4 bg-red-500')
                    ui.label('End Node')
                
                with ui.element().classes('flex items-center gap-2'):
                    ui.element().classes('w-4 h-4 bg-green-100')
                    ui.label('Human Edited')

# ----- EVENT HANDLERS -----
def handle_upload(e):
    """Handle file upload event"""
    # Save the uploaded file
    for file in e.files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file['name'])[1]) as f:
            f.write(file['content'])
            state.image_path = f.name
        ui.notify(f"File uploaded: {file['name']}", type="positive")
        break  # Only process the first file if multiple are uploaded

def set_model(e):
    """Set the AI model"""
    state.model = e.value
    ui.notify(f"Model set to {e.value}", type="info")

def process_image():
    """Process the current image"""
    if not state.image_path:
        ui.notify("Please upload an image first", type="warning")
        return
    
    # Reset state
    state.segments = []
    state.edited_segments = {}
    state.highlighted_segment = None
    state.extraction_complete = False
    
    # Extract segments
    segments = extract_pipe_segments(state.image_path, state.model)
    
    # Update UI
    update_display()
    update_summary()
    update_segment_table()

def set_view_mode(mode):
    """Set the display mode"""
    state.display_mode = mode
    update_display()

def set_line_width(width):
    """Set the line width"""
    state.line_width = int(width)
    update_display()

def export_image():
    """Export the current visualization as an image"""
    if not state.image_path or not state.segments:
        ui.notify("No data to export", type="warning")
        return
    
    try:
        # Generate the visualization
        img_bytes = generate_canvas_image()
        if img_bytes:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{OUTPUT_DIR}/pipe_segments_{timestamp}.png"
            
            with open(output_path, 'wb') as f:
                f.write(img_bytes.getvalue())
            
            ui.notify(f"Image exported to {output_path}", type="positive")
            return output_path
    except Exception as e:
        ui.notify(f"Error exporting image: {e}", type="negative")
        return None

# ----- APP STARTUP -----
# Make sure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Run the app
ui.run(title="Pipe Segment Analyzer", port=8080)
