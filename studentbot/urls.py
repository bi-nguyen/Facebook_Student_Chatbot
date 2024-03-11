from .views import studentresponse
from django.urls import path,include,re_path
urlpatterns = [
    re_path(r'^ab13825e1ec54d36651db11ad98c0acfe217740b310680a/?$',studentresponse,name="studentbot")
]