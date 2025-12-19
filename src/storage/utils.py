import stat
import os
import platform

def is_hidden(filepath):
    # Get just the filename from the full path
    name = os.path.basename(os.path.abspath(filepath))
    
    # 1. Check for Unix/Linux/Mac (Dot convention)
    if name.startswith('.'):
        return True
    
    # 2. Check for Windows (File Attribute)
    if platform.system() == 'Windows':
        try:
            attrs = os.stat(filepath).st_file_attributes
            # Check if the "Hidden" bit is set in the attributes
            return bool(attrs & stat.FILE_ATTRIBUTE_HIDDEN)
        except (AttributeError, OSError):
            # AttributeError: st_file_attributes might not exist on non-Windows
            # OSError: File not found
            return False
            
    return False       