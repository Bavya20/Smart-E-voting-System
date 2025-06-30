from django.urls import path
from . import views
from voting import views


urlpatterns = [
    # Main Pages
    path('', views.welcome, name='welcome'),
    path('vote1', views.vote_view, name='vote1'),
    path('view_votes/', views.view_votes, name='view_votes'),
    path('thank_you1/', views.thank_you1, name='thank_you1'),
    path('register/', views.register, name='register'),
    path('dashboard/', views.voter_dashboard, name='voter_dashboard'),
    path('vote/', views.vote, name='vote'),
    path('thankyou/', views.thank_you, name='thank_you'),
    path('thank-you/', views.thank_you, name='thank_you'),
    # Verification
    path('verify/', views.verify_voter, name='verify_voter'),
    path('check/', views.check, name='check'),

    # Face & Gesture Detection
    path('check_gesture/', views.detect_gesture, name='check_gesture'),  # Optional for debugging
    path('results/', views.results_view, name='results'),

    # Voting & Blockchain Integration
    path('cast_vote/', views.cast_vote, name='cast_vote'),
    path('submit_vote/', views.submit_vote, name='submit_vote'),
    path('get_vote_data/', views.get_vote_data, name='get_vote_data'),
    path('live_results/', views.live_results, name='live_results'), 
    path('results/', views.vote_results, name='vote_results'),
    path('results/', views.results_page, name='results_page'),
    path('results/', views.view_votes, name='view_votes'),
    path('get_live_vote_counts/', views.get_live_vote_counts, name='live_vote_counts'),
    path('verify_face/', views.verify_face_view, name='verify_face'),
    path('verify_face/', views.face_verification, name='verify_face'),
    path('verification_failed/', lambda r: render(r, 'verification_failed.html'), name='verification_failed'),

    # Video Feed Streaming (Live Gesture/Face)
    path('video_feed/', views.video_feed, name='video_feed'),
    path('gesture-feed/', views.gesture_vote_stream, name='gesture_vote_stream'),
    path('vote_status/', views.check_vote_status, name='vote_status'),
    path('verify_face/<int:voter_id>/', views.face_verification, name='verify_face'),
    path('vote/', views.vote_page, name='vote_page'),
    path('vote/', views.cast_vote, name='cast_vote')
]
