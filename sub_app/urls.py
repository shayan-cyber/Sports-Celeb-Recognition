from django.urls import path
from . import views
urlpatterns =[
    path('', views.home, name="home"),
    path('classify_img', views.classify_img, name='classify_img')
]