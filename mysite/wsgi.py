import os
import sys
from pathlib import Path

from django.core.wsgi import get_wsgi_application
from wsgi_sslify import sslify

# Add project directory to the sys.path
path_home = str(Path(__file__).parents[1])
if path_home not in sys.path:
    sys.path.append(path_home)

os.environ['DJANGO_SETTINGS_MODULE'] = 'mysite.settings'

application = sslify(get_wsgi_application())
