from django.urls import path
from django.contrib import admin

from . import views

app_name = 'dashboard'
urlpatterns = [
    path('', views.index, name='index'),
    path('index', views.index, name='index'),
    path('subjects', views.subjects_page, name='subjects'),
    path('utils/', views.utils, name='utils'),
    path('<code>/', views.detail, name='detail'),
    path('utils/<action>', views.utils, name='utils'),
    path('admin/', admin.site.urls),
]
