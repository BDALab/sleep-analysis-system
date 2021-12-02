from django.contrib import admin
from django.urls import path

from . import views

app_name = 'dashboard'
urlpatterns = [
    path('', views.index, name='index'),
    path('index', views.index, name='index'),
    path('subjects', views.subjects_page, name='subjects'),
    path('utils/', views.utils, name='utils'),
    path('<code>/', views.detail, name='detail'),
    path('utils/<action>', views.utils, name='utils'),
    path('<code>/<action>', views.detail_action, name='detail'),
    path('admin/', admin.site.urls),
    path('set_timezone', views.set_timezone, name='set_timezone'),
]
