"""
LangChain compatibility patch
Fixes version compatibility issues with langchain.debug and langchain.verbose
"""

import langchain
import sys

def patch_langchain():
    """Patch langchain to add missing attributes"""
    if not hasattr(langchain, 'debug'):
        langchain.debug = False
        setattr(langchain, 'debug', False)
    
    if not hasattr(langchain, 'verbose'):
        langchain.verbose = False
        setattr(langchain, 'verbose', False)
    
    # Also patch in sys.modules if needed
    if 'langchain' in sys.modules:
        langchain_module = sys.modules['langchain']
        if not hasattr(langchain_module, 'debug'):
            setattr(langchain_module, 'debug', False)
        if not hasattr(langchain_module, 'verbose'):
            setattr(langchain_module, 'verbose', False)

# Auto-patch on import
patch_langchain()
