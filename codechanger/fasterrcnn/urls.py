from django.urls import path
from .views import train_model_view, index

urlpatterns = [
    path('', index, name='index'),
    path('fasterrcnn/', train_model_view, name = 'faster')
]