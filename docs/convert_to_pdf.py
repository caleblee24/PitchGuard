#!/usr/bin/env python3
"""
Convert PitchGuard White Paper to PDF
Usage: python convert_to_pdf.py
"""

import markdown
import pdfkit
import os
from pathlib import Path

def convert_markdown_to_pdf():
    """Convert the white paper markdown to PDF format."""
    
    # File paths
    markdown_file = "docs/PitchGuard_White_Paper_Professional.md"
    output_pdf = "docs/PitchGuard_White_Paper.pdf"
    
    # Check if markdown file exists
    if not os.path.exists(markdown_file):
        print(f"Error: {markdown_file} not found!")
        return False
    
    try:
        # Read markdown content
        with open(markdown_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            md_content,
            extensions=['tables', 'fenced_code', 'codehilite']
        )
        
        # Add CSS styling for professional appearance
        css_style = """
        <style>
        body {
            font-family: 'Times New Roman', serif;
            font-size: 12pt;
            line-height: 1.5;
            margin: 1in;
            color: #333;
        }
        h1 {
            font-size: 18pt;
            font-weight: bold;
            text-align: center;
            margin-top: 2em;
            margin-bottom: 1em;
        }
        h2 {
            font-size: 14pt;
            font-weight: bold;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            page-break-after: avoid;
        }
        h3 {
            font-size: 12pt;
            font-weight: bold;
            margin-top: 1em;
            margin-bottom: 0.5em;
        }
        p {
            margin-bottom: 0.5em;
            text-align: justify;
        }
        code {
            font-family: 'Courier New', monospace;
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
        }
        pre {
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            page-break-inside: avoid;
        }
        ul, ol {
            margin-bottom: 0.5em;
        }
        li {
            margin-bottom: 0.25em;
        }
        .abstract {
            font-style: italic;
            margin: 2em 0;
            padding: 1em;
            background-color: #f9f9f9;
            border-left: 4px solid #007acc;
        }
        </style>
        """
        
        # Create full HTML document
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>PitchGuard: A Machine Learning System for MLB Pitcher Injury Risk Prediction</title>
            {css_style}
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Convert to PDF
        print("Converting markdown to PDF...")
        
        # Try to use wkhtmltopdf if available
        try:
            pdfkit.from_string(full_html, output_pdf)
            print(f"✅ Successfully created: {output_pdf}")
            return True
        except Exception as e:
            print(f"Warning: Could not use wkhtmltopdf: {e}")
            print("Creating HTML file instead...")
            
            # Fallback: save as HTML
            html_file = "docs/PitchGuard_White_Paper.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(full_html)
            print(f"✅ Created HTML file: {html_file}")
            print("You can open this in a browser and print to PDF manually.")
            return False
            
    except Exception as e:
        print(f"Error converting to PDF: {e}")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("Installing required dependencies...")
    os.system("pip install markdown pdfkit")
    print("Note: You may also need to install wkhtmltopdf:")
    print("  macOS: brew install wkhtmltopdf")
    print("  Ubuntu: sudo apt-get install wkhtmltopdf")
    print("  Windows: Download from https://wkhtmltopdf.org/")

if __name__ == "__main__":
    print("PitchGuard White Paper PDF Converter")
    print("=" * 40)
    
    # Check if dependencies are installed
    try:
        import markdown
        import pdfkit
    except ImportError:
        print("Missing dependencies. Installing...")
        install_dependencies()
        exit(1)
    
    # Convert the document
    success = convert_markdown_to_pdf()
    
    if success:
        print("\n🎉 White paper successfully converted to PDF!")
        print("You can now share the PDF with MLB teams and professionals.")
    else:
        print("\n📄 HTML file created. You can open it in a browser and print to PDF.")
    
    print("\nNext steps:")
    print("1. Review the generated PDF/HTML")
    print("2. Customize any content as needed")
    print("3. Share with MLB teams and stakeholders")
    print("4. Use for partnership discussions and presentations")
