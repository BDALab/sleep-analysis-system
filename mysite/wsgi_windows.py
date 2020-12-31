activate_this = 'C:/Users/Administrator/Envs/mysite/Scripts/activate_this.py'
# execfile(activate_this, dict(__file__=activate_this))
exec(open(activate_this).read(),dict(__file__=activate_this))

import os
import sys
import site

# Add the site-packages of the chosen virtualenv to work with
site.addsitedir('C:/Users/Administrator/Envs/mysite/Lib/site-packages')

# Add the app's directory to the PYTHONPATH
sys.path.append('D:/Mikulec/NiceLife/geneactiv-processing-data')
sys.path.append('D:/Mikulec/NiceLife/geneactiv-processing-data/mysite')
sys.path.append('D:/Mikulec/NiceLife/geneactiv-processing-data/dashboard')

os.environ['DJANGO_SETTINGS_MODULE'] = 'mysite.settings'
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mysite.settings")

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()