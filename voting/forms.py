from django import forms
from .models import Voter

class AadhaarVerificationForm(forms.ModelForm):
    class Meta:
        model = Voter
        fields = ['aadhaar_number', 'aadhaar_photo', 'face_photo']

class VoterRegistrationForm(forms.ModelForm):
    class Meta:
        model = Voter
        fields = ['aadhaar_number', 'aadhaar_photo', 'face_photo']
