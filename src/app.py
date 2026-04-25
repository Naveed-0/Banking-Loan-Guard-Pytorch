import torch
import gradio as gr
import logging
from model_def import BankingModel

# Setup logging for production tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = "models/best_banking_model.pth"
DEVICE = torch.device("cpu")

# Load Model with Error Handling
try:
    model = BankingModel(input_size=297)
    # Note: Use weights_only=True for security best practices in global repos
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")

def predict(income, age, ext_1, ext_2, ext_3, loan_amt, years_employed):
    """
    Main inference function mapping 7 UI features to 297 model features.
    """
    try:
        # Initialize sparse feature vector
        input_tensor = torch.zeros(1, 297)
        
        # High-Impact Feature Mapping & Normalization
        input_tensor[0, 0] = income / 100000 
        input_tensor[0, 1] = age / 100
        input_tensor[0, 2] = ext_1
        input_tensor[0, 3] = ext_2
        input_tensor[0, 4] = ext_3
        input_tensor[0, 5] = loan_amt / income if income > 0 else 0
        input_tensor[0, 6] = years_employed / 50 
        
        with torch.no_grad():
            logit = model(input_tensor)
            prob = torch.sigmoid(logit).item()
        
        # Professional 3-Tier Risk Logic
        if prob < 0.15:
            color, label = "#155724", "✅ LOW RISK: AUTO-APPROVE"
            bg = "#d4edda"
        elif prob < 0.35:
            color, label = "#856404", "⚠️ MEDIUM RISK: MANUAL REVIEW"
            bg = "#fff3cd"
        else:
            color, label = "#721c24", "🚩 HIGH RISK: REJECTED"
            bg = "#f8d7da"
            
        verdict_html = f"""
        <div style='color: {color}; background-color: {bg}; padding: 15px; 
        border-radius: 8px; text-align: center; font-weight: bold; font-size: 20px;'>
            {label}
        </div>
        """
        return f"{prob:.2%}", verdict_html

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Error", f"Inference failed: {str(e)}"

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏦 Banking Loan Guard")
    gr.Markdown("### Production-Ready Credit Risk Intelligence Engine")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Applicant Profile")
            income = gr.Slider(0, 500000, value=50000, label="Annual Income ($)")
            age = gr.Slider(18, 95, value=30, label="Age")
            years_exp = gr.Slider(0, 50, value=5, label="Years of Employment")
            loan = gr.Slider(1000, 1000000, value=20000, label="Loan Amount ($)")
            
            gr.Markdown("#### External Bureau Scores")
            s1 = gr.Slider(0, 1, value=0.5, label="Bureau Score A")
            s2 = gr.Slider(0, 1, value=0.5, label="Bureau Score B")
            s3 = gr.Slider(0, 1, value=0.5, label="Bureau Score C")
            
            btn = gr.Button("ANALYZE RISK", variant="primary")
            
        with gr.Column():
            gr.Markdown("#### Risk Decision")
            prob_out = gr.Label(label="Probability of Default")
            verdict_out = gr.HTML()
            
            gr.Markdown("---")
            gr.Markdown("**Note:** This AI assistant is a prototype for financial risk assessment. Decisions should be verified by a credit officer.")

    btn.click(
        fn=predict, 
        inputs=[income, age, s1, s2, s3, loan, years_exp], 
        outputs=[prob_out, verdict_out]
    )

if __name__ == "__main__":
    demo.launch()