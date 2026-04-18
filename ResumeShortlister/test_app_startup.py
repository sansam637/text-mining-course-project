"""Test Flask app startup"""
from app import create_app

app = create_app('development')
print('✓ Flask app created successfully')
print(f'  Upload folder: {app.config["UPLOAD_FOLDER"]}')
print(f'  Job descriptions folder: {app.config["JOB_DESCRIPTIONS_FOLDER"]}')
print(f'  Allowed extensions: {app.config["ALLOWED_EXTENSIONS"]}')
print('\n✓ All systems ready! The resume processing feature should work now.')
