# from django.urls import path
# from .views import predict

# urlpatterns = [
#     path("", predict, name="predict"),
# ]



from django.urls import path
from .views import landing, realtime, analysis_page, predict

urlpatterns = [
    path("", landing, name="landing"),            # Landing page
    path("realtime/", realtime, name="realtime"), # Real-time detection
    path("analysis/", analysis_page, name="analysis"), # File upload analysis
    path("predict/", predict, name="predict"),    # API endpoint
]
