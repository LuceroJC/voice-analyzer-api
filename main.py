"""
VoiceFlow Tools - Voice Analysis API
FastAPI backend for acoustic voice analysis
Author: Dr. Jorge C. Lucero
Version: 1.1.0 - Memory Optimized
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
import parselmouth
from parselmouth.praat import call
import io
import base64
from typing import Dict, Any, Optional, Union, List
import tempfile
import os
from pydantic import BaseModel
from datetime import datetime
import logging
import traceback
from pydub import AudioSegment
import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning, module='pydub')

# PDF generation imports
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Phonalab Voice Analyzer API",
    description="Professional acoustic voice analysis for clinicians",
    version="1.1.0"
)

# Configure CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MEMORY OPTIMIZATION SETTINGS =====
MAX_DURATION_SECONDS = 30  # Analyze first 30 seconds only
TARGET_SAMPLE_RATE = 16000  # Downsample to 16kHz (sufficient for voice analysis)
# ========================================

class AnalysisResult(BaseModel):
    """Response model for voice analysis"""
    f0: Dict[str, Union[float, str, List[float]]]
    jitter: Dict[str, Union[float, str]]
    shimmer: Dict[str, Union[float, str]]
    hnr: Dict[str, Union[float, str]]
    cpp: Dict[str, Union[float, str]]
    interpretation: Dict[str, Any]
    metadata: Dict[str, Any]

class FeedbackSubmission(BaseModel):
    """Model for user feedback"""
    rating: int
    comment: Optional[str] = None
    feature_request: Optional[str] = None
    user_type: Optional[str] = None
    timestamp: Optional[str] = None

class PDFRequest(BaseModel):
    """Model for PDF generation request"""
    analysis_results: AnalysisResult
    patient_info: Optional[Dict[str, str]] = None

# In-memory feedback storage (replace with database in production)
feedback_storage: List[Dict] = []

@app.get("/")
async def root():
    """API health check and info"""
    return {
        "status": "active",
        "service": "Phonalab Voice Analyzer",
        "version": "1.1.0",
        "author": "Dr. Jorge C. Lucero",
        "description": "Professional acoustic voice analysis API",
        "max_analysis_duration": f"{MAX_DURATION_SECONDS} seconds",
        "optimized_sample_rate": f"{TARGET_SAMPLE_RATE} Hz"
    }

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_voice(file: UploadFile = File(...)):
    """
    Analyze voice recording and return acoustic parameters
    
    Parameters:
    - file: Audio file (WAV, MP3, M4A, CAF, etc.)
    
    Returns:
    - JSON with F0, jitter, shimmer, HNR, and clinical interpretation
    
    Note: For memory efficiency, files longer than 30 seconds will be automatically
    trimmed to the first 30 seconds for analysis.
    """
    
    # Size limit (50MB)
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be less than 50MB")
    
    tmp_input_path = None
    tmp_wav_path = None
    
    try:
        logger.info(f"Received file: {file.filename}, type: {file.content_type}, size: {len(contents)} bytes")
        
        # Determine file extension - try multiple methods
        file_extension = os.path.splitext(file.filename)[1].lower() if file.filename else ''
        
        # If no extension from filename, guess from content type
        if not file_extension or file_extension == '.':
            content_type_map = {
                'audio/mp4': '.m4a',
                'audio/x-m4a': '.m4a',
                'audio/mpeg': '.mp3',
                'audio/mp3': '.mp3',
                'audio/wav': '.wav',
                'audio/wave': '.wav',
                'audio/x-wav': '.wav',
                'audio/x-caf': '.caf',
                'audio/webm': '.webm',
                'audio/ogg': '.ogg'
            }
            file_extension = content_type_map.get(file.content_type, '.m4a')
            logger.info(f"No extension found, using: {file_extension} based on content type")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(contents)
            tmp_input_path = tmp_file.name
        
        logger.info(f"Saved to temp file: {tmp_input_path}")
        
        # Convert to WAV with optimization
        logger.info(f"Converting {file_extension} to optimized WAV...")
        try:
            # Load audio with pydub
            audio = AudioSegment.from_file(tmp_input_path)
            
            # Store original duration before trimming
            original_duration = len(audio) / 1000.0
            
            # Convert to mono
            if audio.channels > 1:
                logger.info("Converting to mono...")
                audio = audio.set_channels(1)
            
            # Downsample to target rate for memory efficiency
            logger.info(f"Resampling to {TARGET_SAMPLE_RATE}Hz for optimal memory usage...")
            audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
            
            # Trim to max duration
            if len(audio) > MAX_DURATION_SECONDS * 1000:
                logger.info(f"Trimming audio from {original_duration:.1f}s to {MAX_DURATION_SECONDS}s")
                audio = audio[:MAX_DURATION_SECONDS * 1000]
            
            # Export as WAV
            tmp_wav_path = tmp_input_path.replace(file_extension, '.wav')
            audio.export(
                tmp_wav_path, 
                format='wav',
                parameters=["-ac", "1"]
            )
            logger.info(f"Converted to WAV: {tmp_wav_path}")
            
            # Use the WAV file for analysis
            analysis_path = tmp_wav_path
            
        except Exception as e:
            logger.error(f"Conversion error: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=400, 
                detail=f"Could not process audio file. Please ensure it's a valid voice recording. Supported formats: M4A, MP3, WAV, CAF. Error: {str(e)}"
            )
        
        # Load with Parselmouth for acoustic analysis
        logger.info("Loading audio with Parselmouth...")
        sound = parselmouth.Sound(analysis_path)
        
        # Get duration and sample rate from Parselmouth
        duration = sound.duration
        sample_rate = int(sound.sampling_frequency)
        
        logger.info(f"Audio loaded: {duration:.2f}s at {sample_rate}Hz")
        
        # Check if audio is long enough for analysis
        if duration < 0.5:
            raise HTTPException(
                status_code=400,
                detail="Recording too short. Please provide at least 0.5 seconds of audio for analysis."
            )
        
        logger.info("Analyzing F0...")
        f0_data = analyze_f0(sound)
        
        logger.info("Analyzing jitter...")
        jitter_data = analyze_jitter(sound)
        
        logger.info("Analyzing shimmer...")
        shimmer_data = analyze_shimmer(sound)
        
        logger.info("Analyzing HNR...")
        hnr_data = analyze_hnr(sound)
        
        logger.info("Analyzing CPP (memory-optimized)...")
        cpp_data = analyze_cpp_efficient(sound)
        
        logger.info("Generating interpretation...")
        interpretation = generate_interpretation(f0_data, jitter_data, shimmer_data, hnr_data, cpp_data)
        
        # Clean up temp files
        if tmp_input_path and os.path.exists(tmp_input_path):
            os.unlink(tmp_input_path)
        if tmp_wav_path and os.path.exists(tmp_wav_path):
            os.unlink(tmp_wav_path)
            
        logger.info("Analysis completed successfully")
        
        # Prepare metadata
        metadata = {
            "filename": file.filename,
            "duration": round(duration, 2),
            "sample_rate": int(sample_rate),
            "analysis_date": datetime.now().isoformat(),
            "original_format": file_extension,
            "analyzed_duration": round(min(duration, MAX_DURATION_SECONDS), 2)
        }
        
        # Add note if file was trimmed
        if original_duration > MAX_DURATION_SECONDS:
            metadata["note"] = f"Original file was {original_duration:.1f}s. Analysis performed on first {MAX_DURATION_SECONDS}s for memory optimization."
        
        # Extract F0 contour
        f0_values = []
        for frame in sound.to_pitch().selected_array['frequency']:
            if frame > 0:  # Voiced frames only
                f0_values.append(frame)
                
        return AnalysisResult(
            f0=f0_data,
            jitter=jitter_data,
            shimmer=shimmer_data,
            hnr=hnr_data,
            cpp=cpp_data,
            interpretation=interpretation,
            metadata=metadata
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        # Clean up temp files if they exist
        if tmp_input_path and os.path.exists(tmp_input_path):
            try:
                os.unlink(tmp_input_path)
            except:
                pass
        if tmp_wav_path and os.path.exists(tmp_wav_path):
            try:
                os.unlink(tmp_wav_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
    
@app.post("/generate-pdf")
async def generate_pdf_report(request: PDFRequest):
    """
    Generate PDF report from analysis results
    
    Parameters:
    - request: PDFRequest with analysis results and optional patient info
    
    Returns:
    - PDF file as downloadable stream
    """
    try:
        logger.info("Generating PDF report...")
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        # Container for PDF elements
        elements = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#6366f1'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#4f46e5'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        elements.append(Paragraph("Voice Analysis Report", title_style))
        elements.append(Paragraph("Phonalab", 
            ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=12, 
                          textColor=colors.HexColor('#6b7280'), alignment=TA_CENTER)))
        elements.append(Spacer(1, 0.2*inch))
        
        # Metadata
        metadata = request.analysis_results.metadata
        elements.append(Paragraph("Recording Information", heading_style))

        info_data = [
            ["Filename:", metadata.get('filename', 'N/A')],
            ["Duration:", f"{metadata.get('duration', 0)} seconds"],
            ["Sample Rate:", f"{metadata.get('sample_rate', 0)} Hz"],
            ["Analysis Date:", metadata.get('analysis_date', datetime.now().isoformat())[:19].replace('T', ' ')],
        ]

        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(info_table)

        # Add patient info if provided
        if request.patient_info and any(request.patient_info.values() if isinstance(request.patient_info, dict) else []):
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph("Patient Information", heading_style))
            patient_data = [[k.replace('_', ' ').title() + ':', v] for k, v in request.patient_info.items() if v]
            if patient_data:
                patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
                patient_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                    ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ]))
                elements.append(patient_table)

        elements.append(Spacer(1, 0.3*inch))
        
        # Acoustic Parameters
        elements.append(Paragraph("Acoustic Measurements", heading_style))
        
        results = request.analysis_results
        
        # F0
        f0_data = [
            ["Parameter", "Value", "Status"],
            ["Mean F0", f"{results.f0.get('mean', 0):.1f} Hz", results.f0.get('status', 'N/A')],
            ["F0 Range", f"{results.f0.get('min', 0):.1f} - {results.f0.get('max', 0):.1f} Hz", ""],
            ["F0 Std Dev", f"{results.f0.get('std', 0):.1f} Hz", ""],
        ]
        
        # Jitter
        jitter_data = [
            ["Jitter (Local)", f"{results.jitter.get('local', 0):.4f}", results.jitter.get('status', 'N/A')],
            ["Jitter (%)", f"{results.jitter.get('percent', 0):.2f}%", ""],
            ["RAP", f"{results.jitter.get('rap', 0):.4f}", ""],
        ]
        
        # Shimmer
        shimmer_data = [
            ["Shimmer (Local)", f"{results.shimmer.get('local', 0):.4f}", results.shimmer.get('status', 'N/A')],
            ["Shimmer (%)", f"{results.shimmer.get('percent', 0):.2f}%", ""],
            ["APQ3", f"{results.shimmer.get('apq3', 0):.4f}", ""],
        ]
        
        # HNR
        hnr_data = [
            ["Mean HNR", f"{results.hnr.get('mean', 0):.1f} dB", results.hnr.get('status', 'N/A')],
            ["HNR Range", f"{results.hnr.get('min', 0):.1f} - {results.hnr.get('max', 0):.1f} dB", ""],
        ]
        
        # CPP
        cpp_data = [
            ["CPP (Smoothed)", f"{results.cpp.get('value', 0):.2f} dB", results.cpp.get('status', 'N/A')],
        ]
        
        # Combine all measurements
        all_data = [["Parameter", "Value", "Status"]] + f0_data[1:] + jitter_data + shimmer_data + hnr_data + cpp_data
        
        measurements_table = Table(all_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        measurements_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        elements.append(measurements_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Clinical Interpretation
        elements.append(Paragraph("Clinical Interpretation", heading_style))
        
        interpretation = results.interpretation
        elements.append(Paragraph(
            f"<b>Overall Assessment:</b> {interpretation.get('overall_assessment', 'N/A')}", 
            styles['Normal']
        ))
        elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Paragraph(
            f"<b>Severity:</b> {interpretation.get('severity', 'N/A').upper()}", 
            styles['Normal']
        ))
        elements.append(Spacer(1, 0.2*inch))
        
        # Concerns
        if interpretation.get('concerns'):
            elements.append(Paragraph("<b>Clinical Concerns:</b>", styles['Normal']))
            for concern in interpretation['concerns']:
                elements.append(Paragraph(f"• {concern}", styles['Normal']))
            elements.append(Spacer(1, 0.2*inch))
        
        # Recommendations
        if interpretation.get('recommendations'):
            elements.append(Paragraph("<b>Recommendations:</b>", styles['Normal']))
            for rec in interpretation['recommendations']:
                elements.append(Paragraph(f"• {rec}", styles['Normal']))
            elements.append(Spacer(1, 0.2*inch))
        
        # Clinical Action
        elements.append(Paragraph(
            f"<b>Suggested Action:</b> {interpretation.get('clinical_action', 'N/A')}", 
            styles['Normal']
        ))
        
        # Footer
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph(
            "<i>This report was generated by Phonalab - Professional Voice Analysis Tools</i>",
            ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
        ))
        elements.append(Paragraph(
            "<i>Created by Dr. Jorge C. Lucero | For clinical correlation and professional use only</i>",
            ParagraphStyle('Footer2', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
        ))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        logger.info("PDF generated successfully")
        
        # Return as downloadable file
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=voice_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
        )
        
    except Exception as e:
        logger.error(f"PDF generation error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackSubmission):
    """
    Submit user feedback
    
    Parameters:
    - feedback: FeedbackSubmission with rating and optional comments
    
    Returns:
    - Confirmation message
    """
    try:
        # Add timestamp if not provided
        if not feedback.timestamp:
            feedback.timestamp = datetime.now().isoformat()
        
        # Store feedback (in production, save to database)
        feedback_data = feedback.dict()
        feedback_storage.append(feedback_data)
        
        logger.info(f"Feedback received: Rating {feedback.rating}/5")
        if feedback.comment:
            logger.info(f"Comment: {feedback.comment}")
        if feedback.feature_request:
            logger.info(f"Feature request: {feedback.feature_request}")
        
        return {
            "status": "success",
            "message": "Thank you for your feedback!",
            "feedback_id": len(feedback_storage)
        }
        
    except Exception as e:
        logger.error(f"Feedback submission error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")

@app.get("/feedback/stats")
async def get_feedback_stats():
    """
    Get feedback statistics (for admin use)
    
    Returns:
    - Summary of feedback data
    """
    if not feedback_storage:
        return {
            "total_feedback": 0,
            "average_rating": 0,
            "ratings_distribution": {}
        }
    
    ratings = [f['rating'] for f in feedback_storage]
    
    return {
        "total_feedback": len(feedback_storage),
        "average_rating": round(sum(ratings) / len(ratings), 2),
        "ratings_distribution": {
            str(i): ratings.count(i) for i in range(1, 6)
        },
        "recent_comments": [
            {
                "rating": f['rating'],
                "comment": f.get('comment', ''),
                "feature_request": f.get('feature_request', ''),
                "timestamp": f.get('timestamp', '')
            }
            for f in feedback_storage[-10:]
        ]
    }

# ===== ANALYSIS HELPER FUNCTIONS =====

def analyze_f0(sound: parselmouth.Sound) -> Dict[str, Union[float, str]]:
    """Extract fundamental frequency statistics"""
    try:
        pitch = sound.to_pitch()
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0]
        
        if len(f0_values) == 0:
            return {
                "mean": 0,
                "std": 0,
                "min": 0,
                "max": 0,
                "status": "Could not detect pitch"
            }
        f0_contour = f0_values[::10].tolist()
        return {
            "mean": float(np.mean(f0_values)),
            "std": float(np.std(f0_values)),
            "min": float(np.min(f0_values)),
            "max": float(np.max(f0_values)),
            "percentile_5": float(np.percentile(f0_values, 5)),
            "percentile_95": float(np.percentile(f0_values, 5)),
            "status": get_f0_status(float(np.mean(f0_values)))
        }
    except Exception as e:
        return {"error": str(e), "status": "Analysis failed"}
    

def analyze_jitter(sound: parselmouth.Sound) -> Dict[str, Union[float, str]]:
    """Calculate jitter parameters"""
    try:
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_percent = local_jitter * 100
        rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        
        return {
            "local": float(local_jitter),
            "percent": float(jitter_percent),
            "rap": float(rap),
            "status": get_jitter_status(jitter_percent)
        }
    except Exception as e:
        return {"error": str(e), "status": "Analysis failed"}

def analyze_shimmer(sound: parselmouth.Sound) -> Dict[str, Union[float, str]]:
    """Calculate shimmer parameters"""
    try:
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        local_shimmer = call([sound, point_process], "Get shimmer (local)", 
                            0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_percent = local_shimmer * 100
        apq3 = call([sound, point_process], "Get shimmer (apq3)", 
                   0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        return {
            "local": float(local_shimmer),
            "percent": float(shimmer_percent),
            "apq3": float(apq3),
            "status": get_shimmer_status(shimmer_percent)
        }
    except Exception as e:
        return {"error": str(e), "status": "Analysis failed"}

def analyze_hnr(sound: parselmouth.Sound) -> Dict[str, Union[float, str]]:
    """Calculate Harmonics-to-Noise Ratio"""
    try:
        harmonicity = sound.to_harmonicity()
        hnr_values = harmonicity.values[harmonicity.values != -200]
        
        if len(hnr_values) == 0:
            return {
                "mean": 0,
                "std": 0,
                "status": "Could not calculate HNR"
            }
        
        mean_hnr = float(np.mean(hnr_values))
        
        return {
            "mean": mean_hnr,
            "std": float(np.std(hnr_values)),
            "min": float(np.min(hnr_values)),
            "max": float(np.max(hnr_values)),
            "status": get_hnr_status(mean_hnr)
        }
    except Exception as e:
        return {"error": str(e), "status": "Analysis failed"}

def analyze_cpp_efficient(sound: parselmouth.Sound) -> Dict[str, Union[float, str]]:
    """
    Memory-efficient CPP analysis using chunking
    This is the OPTIMIZED version that processes in chunks to reduce memory usage
    """
    try:
        duration = sound.get_total_duration()
        chunk_size = 5.0  # 5-second chunks
        cpp_values = []
        
        num_chunks = min(int(duration / chunk_size) + 1, 6)  # Max 6 chunks
        
        for i in range(num_chunks):
            start_time = i * chunk_size
            end_time = min(start_time + chunk_size, duration)
            
            if end_time - start_time < 1.0:  # Skip chunks shorter than 1 second
                continue
            
            try:
                # Extract chunk
                chunk = call(sound, "Extract part", start_time, end_time, "rectangular", 1.0, "no")
                
                # Calculate CPP for chunk
                power_cepstrogram = call(chunk, "To PowerCepstrogram", 60.0, 0.002, 5000.0, 50.0)
                cpp_chunk = call(power_cepstrogram, "Get CPPS", "yes", 0.01, 0.001, 60.0, 333.3, 0.05, "Parabolic", 0.001, 0.0, "Straight", "Robust")
                
                if cpp_chunk is not None:
                    cpp_values.append(float(cpp_chunk))
                
                # Explicit cleanup
                del chunk
                del power_cepstrogram
                
            except Exception as chunk_error:
                logger.warning(f"CPP chunk {i} failed: {chunk_error}")
                continue
        
        # Return mean CPP
        if cpp_values:
            mean_cpp = sum(cpp_values) / len(cpp_values)
            return {
                "value": float(mean_cpp),
                "status": get_cpp_status(float(mean_cpp))
            }
        else:
            return {"value": 0.0, "status": "Could not calculate CPP"}
        
    except Exception as e:
        logger.error(f"CPP analysis error: {str(e)}")
        logger.error(traceback.format_exc())
        return {"value": 0.0, "status": "Could not calculate CPP"}

def get_f0_status(f0_mean: float) -> str:
    """Interpret F0 value"""
    if f0_mean < 85:
        return "Very low - possible pathology"
    elif 85 <= f0_mean < 155:
        return "Normal range (adult male)"
    elif 155 <= f0_mean < 255:
        return "Normal range (adult female)"
    else:
        return "High - possible tension or pediatric"

def get_jitter_status(jitter_percent: float) -> str:
    """Interpret jitter value"""
    if jitter_percent < 1.0:
        return "Normal"
    elif jitter_percent < 2.0:
        return "Slightly elevated"
    else:
        return "Elevated - suggests voice disorder"

def get_shimmer_status(shimmer_percent: float) -> str:
    """Interpret shimmer value"""
    if shimmer_percent < 3.0:
        return "Normal"
    elif shimmer_percent < 5.0:
        return "Slightly elevated"
    else:
        return "Elevated - suggests voice disorder"

def get_hnr_status(hnr_mean: float) -> str:
    """Interpret HNR value"""
    if hnr_mean > 20:
        return "Good voice quality"
    elif hnr_mean > 15:
        return "Mild noise component"
    else:
        return "Significant noise - possible disorder"

def get_cpp_status(cpp_value: float) -> str:
    """Interpret CPP value"""
    if cpp_value > 6.0:
        return "Good voice quality"
    elif cpp_value > 5.0:
        return "Mild voice quality reduction"
    elif cpp_value > 3.0:
        return "Moderate dysphonia"
    else:
        return "Severe dysphonia"

def generate_interpretation(f0: Dict, jitter: Dict, shimmer: Dict, hnr: Dict, cpp: Dict) -> Dict:
    """Generate clinical interpretation of results"""
    concerns = []
    recommendations = []
    
    if 'percent' in jitter and jitter['percent'] > 1.0:
        concerns.append("Elevated jitter (pitch instability)")
        recommendations.append("Consider vocal fold examination")
    
    if 'percent' in shimmer and shimmer['percent'] > 3.0:
        concerns.append("Elevated shimmer (amplitude instability)")
        recommendations.append("Assess for vocal fold lesions")
    
    if 'mean' in hnr and hnr['mean'] < 20:
        concerns.append("Reduced harmonics-to-noise ratio")
        recommendations.append("Evaluate for breathiness or roughness")
    
    if 'value' in cpp and cpp['value'] < 5.0:
        concerns.append("Reduced cepstral peak prominence")
        recommendations.append("Consider comprehensive voice evaluation")
    
    if 'mean' in f0:
        if f0['mean'] < 85 or f0['mean'] > 300:
            concerns.append("Abnormal fundamental frequency")
            recommendations.append("Check for structural or functional issues")
    
    if len(concerns) == 0:
        overall = "Voice parameters within normal limits"
        severity = "normal"
    elif len(concerns) == 1:
        overall = "Mild voice quality concerns noted"
        severity = "mild"
    elif len(concerns) == 2:
        overall = "Moderate voice disorder indicators present"
        severity = "moderate"
    else:
        overall = "Multiple parameters suggest significant voice disorder"
        severity = "severe"
    
    return {
        "overall_assessment": overall,
        "severity": severity,
        "concerns": concerns,
        "recommendations": recommendations,
        "clinical_action": get_clinical_action(severity)
    }

def get_clinical_action(severity: str) -> str:
    """Suggest clinical action based on severity"""
    actions = {
        "normal": "Continue routine monitoring",
        "mild": "Monitor and consider voice hygiene education",
        "moderate": "Recommend comprehensive voice evaluation",
        "severe": "Urgent referral to otolaryngology recommended"
    }
    return actions.get(severity, "Clinical correlation recommended")

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "service": "voice-analyzer", "version": "1.1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)