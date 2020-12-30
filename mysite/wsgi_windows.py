import os
import site
import sys

# Add the site-packages of the chosen virtualenv to work with
site.addsitedir('D:/ProgramData/Miniconda3/envs/geneactiv-processing-data/Lib/site-packages')

# Add the app's directory to the PYTHONPATH
sys.path.append('E:/geneactiv-processing-data/www/mysite')
sys.path.append('E:/geneactiv-processing-data/www/mysite/mysite')

os.environ['DJANGO_SETTINGS_MODULE'] = 'mysite.settings'
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "my_application.settings")

from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()
