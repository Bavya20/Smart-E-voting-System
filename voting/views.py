import cv2
import csv
import json
import dlib
import numpy as np
import os
import threading
import mediapipe as mp
import face_recognition
import traceback
from datetime import datetime
from PIL import Image
import tempfile
from django.http import HttpResponse
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from django.db import IntegrityError
from django.core.files.storage import FileSystemStorage
from web3 import Web3
from .models import Voter, Party, Vote
from django.views.decorators.csrf import csrf_exempt
from .blockchain import web3, contract
from .blockchain_config import contract, web3  

global vote_submitted
vote_submitted = False
# =============================
# Global Config & File Paths
# =============================
BASE_DIR = settings.BASE_DIR
MODEL_PATH = os.path.join(BASE_DIR, "voting", "shape_predictor_68_face_landmarks.dat")
CONTRACT_PATH = os.path.join(BASE_DIR, "contract_data.json")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"⚠️ Model file not found: {MODEL_PATH}. Please download it.")
if not os.path.exists(CONTRACT_PATH):
    raise RuntimeError(f"⚠️ Smart contract file not found: {CONTRACT_PATH}. Ensure blockchain is set up.")

WEB3_PROVIDER = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))
contract = None
try:
    with open(CONTRACT_PATH, "r") as f:
        contract_data = json.load(f)
    contract = web3.eth.contract(address=contract_data["address"], abi=contract_data["abi"])
except (FileNotFoundError, json.JSONDecodeError):
    print("⚠️ Warning: Blockchain contract not loaded.")

# =============================
# Mediapipe & Dlib Setup
# =============================
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh()
hands = mp_hands.Hands()
face_detector = dlib.get_frontal_face_detector()

# =============================
# Head Nod Detection Variables
# =============================
NOD_THRESHOLD = 15
previous_nod_angle = None
nod_detected = False
last_selected_party = None

# =============================
# Helper Functions
# =============================
def detect_head_nod(face_landmarks):
    global previous_nod_angle, nod_detected
    chin = face_landmarks.landmark[152]
    forehead = face_landmarks.landmark[10]
    chin_y = chin.y
    forehead_y = forehead.y
    angle = np.degrees(np.arctan2(chin_y - forehead_y, 1))

    if previous_nod_angle is not None:
        angle_diff = abs(angle - previous_nod_angle)
        if angle_diff > NOD_THRESHOLD:
            nod_detected = True
    previous_nod_angle = angle


def process_gesture_frame(frame):
    global nod_detected, last_selected_party
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(imgRGB)
    results_face = face_mesh.process(imgRGB)

    if results_hands.multi_hand_landmarks:
        for handLms in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            gesture_code = detect_gesture(frame)
            if gesture_code:
                last_selected_party = gesture_code

    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            detect_head_nod(face_landmarks)
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

    label = f"Selected Party: {last_selected_party}" if last_selected_party else "No Selection"
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if nod_detected and last_selected_party:
        nod_detected = False
        return frame, "Confirm"

    return frame, last_selected_party or "No Selection"

# =============================
# Views
# =============================
def welcome(request):
    return render(request, 'index.html')

def thank_you(request):
    return render(request, 'thankyou.html')

def vote(request):
    parties = ["BJP", "Congress", "AAP", "TMC", "ShivSena", "BSP", "SP", "CPI", "NOTA"]
    return render(request, 'vote.html', {'parties': parties})

def vote_results(request):
    return render(request, 'results.html')

def check(request):
    return render(request, 'check.html')

from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Voter
import pytesseract
from PIL import Image
import io
import re

def extract_text_from_image(image_file):
    try:
        img = Image.open(image_file)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print("OCR Error:", e)
        return ""

def register(request):
    if request.method == 'POST':
        # Get POST data and strip leading/trailing spaces
        full_name = request.POST.get('full_name', '').strip()
        father_name = request.POST.get('father_name', '').strip()
        age = request.POST.get('age', '').strip()
        aadhaar_number = request.POST.get('aadhaar_number', '').strip()
        house_no = request.POST.get('house_no', '').strip()
        street = request.POST.get('street', '').strip()
        village_town = request.POST.get('village_town', '').strip()
        ward = request.POST.get('ward', '').strip()
        mandal = request.POST.get('mandal', '').strip()
        district = request.POST.get('district', '').strip()
        pincode = request.POST.get('pincode', '').strip()
        state = request.POST.get('state', '').strip()
        is_disabled = request.POST.get('is_disabled', 'no').strip()
        vote_type = request.POST.get('vote_type', '').strip()

        aadhaar_photo = request.FILES.get('aadhaar_photo')
        face_photo = request.FILES.get('face_photo')

        errors = []

        # Validations
        if len(full_name) < 3:
            errors.append("Full Name must be at least 3 characters.")

        if len(father_name) < 3:
            errors.append("Father's Name must be at least 3 characters.")

        try:
            age = int(age)
            if age < 18:
                errors.append("Age must be at least 18.")
        except:
            errors.append("Age must be a valid number.")

        if not aadhaar_number.isdigit() or len(aadhaar_number) != 12:
            errors.append("Aadhaar number must be exactly 12 digits.")

        if not pincode.isdigit() or len(pincode) != 6:
            errors.append("Pincode must be exactly 6 digits.")

        if not aadhaar_photo:
            errors.append("Aadhaar photo is required.")
        else:
            # Extract Aadhaar number using OCR
            ocr_text = extract_text_from_image(aadhaar_photo)
            numbers = re.findall(r'\b\d{12}\b', ocr_text)
            if aadhaar_number not in numbers:
                errors.append("Aadhaar number does not match the one in the uploaded photo.")

        if not face_photo:
            errors.append("Face photo is required.")

        # If any errors, show them
        if errors:
            for error in errors:
                messages.error(request, error)
            return render(request, 'register.html')

        # Save to database
        Voter.objects.create(
            full_name=full_name,
            father_name=father_name,
            age=age,
            aadhaar_number=aadhaar_number,
            house_no=house_no,
            street=street,
            village_town=village_town,
            ward=ward,
            mandal=mandal,
            district=district,
            pincode=pincode,
            state=state,
            is_disabled=is_disabled,
            vote_type=vote_type,
            aadhaar_photo=aadhaar_photo,
            face_photo=face_photo
        )

        messages.success(request, "Voter registered successfully!")
        return redirect('dashboard')

    return render(request, 'register.html')


def dashboard(request):
    voters = Voter.objects.all()
    return render(request, 'dashboard.html', {'voters': voters})

from django.shortcuts import render, redirect
from .models import Voter

def verify_voter(request):
    if request.method == 'POST':
        aadhar = request.POST.get('aadhaar_number')

        try:
            voter = Voter.objects.get(aadhaar_number=aadhar)

            if not voter.face_photo:
                return render(request, 'verification_failed.html', {'error': 'No Aadhaar face photo found.'})

            # Save Aadhaar to session if needed
            request.session['verified_aadhar'] = aadhar
            return render(request, 'face_verified.html', {'voter': voter})

        except Voter.DoesNotExist:
            return render(request, 'verification_failed.html', {'error': 'Aadhaar not found.'})

    return redirect('welcome')


# MediaPipe Setup
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
mpDraw = mp.solutions.drawing_utils

gesture_party_mapping = {
    "00000": "BJP",
    "10000": "Congress",
    "11000": "AAP",
    "11100": "TMC",
    "11110": "ShivSena",
    "11111": "BSP",
    "01000": "SP",
    "01100": "CPI",
    "01111": "NOTA"
}

virtual_keyboard = list("ABCDEFGHIJ") + list("KLMNOPQRST")

vote_submitted = False  # Global flag

# Generator Function
def gen_combined_feed(request):
    global vote_submitted
    cap = cv2.VideoCapture(0)
    prev_nod_y = None
    selected_party = "No Selection"
    vote_confirmation_text = ""

    # Fetch voter
    voter = Voter.objects.filter(has_voted=False).first()

    # Get start time from session
    start_time_str = request.session.get("start_time")

    while True:
        success, frame = cap.read()
        if not success:
            break

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Gesture Detection
        results_hands = hands.process(imgRGB)
        if results_hands.multi_hand_landmarks:
            for handLms in results_hands.multi_hand_landmarks:
                fingers = []
                for i in range(5):
                    tip_id = [4, 8, 12, 16, 20][i]
                    if handLms.landmark[tip_id].y < handLms.landmark[tip_id - 2].y:
                        fingers.append("1")
                    else:
                        fingers.append("0")
                gesture_code = "".join(fingers)
                selected_party = gesture_party_mapping.get(gesture_code, "Unknown Gesture")
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

        # Nod Detection
        results_face = faceMesh.process(imgRGB)
        if results_face.multi_face_landmarks:
            for faceLms in results_face.multi_face_landmarks:
                nod_y = faceLms.landmark[10].y

                if prev_nod_y is not None and abs(nod_y - prev_nod_y) > 0.02 and not vote_submitted:
                    vote_submitted = True
                    vote_confirmation_text = f"✅ Vote Submitted for: {selected_party}"
                    print("***********************************************************")
                    print(vote_confirmation_text)

                    party = selected_party
                    if voter:
                        voter.has_voted = True
                        voter.voted_party = party
                        voter.save()
                    else:
                        print("No unvoted users found.")

                    try:
                        # Calculate time taken
                        if start_time_str:
                            start_time = datetime.fromisoformat(start_time_str)
                            time_taken = (datetime.now() - start_time).total_seconds()
                        else:
                            time_taken = "N/A"

                        file_path = "voters.csv"
                        file_exists = os.path.isfile(file_path)
                        is_empty = not file_exists or os.stat(file_path).st_size == 0  # Check if file is empty

                        
                        with open(file_path, "a", newline='') as f:
                            writer = csv.writer(f)

                            # Write headers only if file is new or empty
                            if is_empty:
                                writer.writerow([
                                    "Aadhaar Number",
                                    "Full Name",
                                    "Age",
                                    "Voted Party",
                                    "Timestamp",
                                    "Time Taken"
                                ])

                            # Write the vote details
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            writer.writerow([
                                voter.aadhaar_number,
                                voter.full_name,
                                voter.age,
                                voter.voted_party,
                                timestamp,
                                f"{time_taken} seconds"
                            ])

                        print("✅ Vote successfully written to CSV with headers")
                    except Exception as e:
                        print("❌ Error writing to CSV:", e)

                prev_nod_y = nod_y
                mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_CONTOURS)

        # UI Overlay
        cv2.putText(frame, f"Selected: {selected_party}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, vote_confirmation_text, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        for i, letter in enumerate(virtual_keyboard):
            x_pos, y_pos = 50 + (i % 10) * 50, 150 + (i // 10) * 50
            cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 40, y_pos + 40), (255, 0, 0), 2)
            cv2.putText(frame, letter, (x_pos + 10, y_pos + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Your Original View
def gesture_vote_stream(request):
    # Store the voting start time in session
    request.session['start_time'] = datetime.now().isoformat()
    return StreamingHttpResponse(
        gen_combined_feed(request),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


@csrf_exempt
def cast_vote(request):
    print("✅ cast_vote hit.")
    try:
        if request.method != "POST":
            return JsonResponse({"status": "error", "message": "Invalid request method"})

        data = json.loads(request.body)
        selected_party = data.get("party")
        source = data.get("source", "button")  # default to "button"
        voter_id = data.get("voter_id") or request.session.get("voter_id")

        print(f"✅ cast_vote hit. Party: {selected_party}, Source: {source}, Voter ID: {voter_id}")

        if not voter_id:
            return JsonResponse({"status": "error", "message": "Voter ID missing."})

        voter = Voter.objects.get(id=voter_id)

        if voter.has_voted:
            return JsonResponse({"status": "error", "message": "Already voted"})

        voter.has_voted = True
        voter.voted_party = selected_party
        voter.save()

        with open("votes.csv", "a") as f:
            f.write(f"{voter.aadhar},{selected_party},{datetime.now()}\n")

        print("✅ Vote successfully written to CSV")

        return JsonResponse({"status": "success", "message": "Vote submitted"})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"status": "error", "message": str(e)})


def view_votes(request):
    votes = []
    filename = 'votes.csv'  # Adjust path if stored in a subfolder

    try:
        with open(filename, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                votes.append(row)
    except FileNotFoundError:
        votes = []

    return render(request, 'voting/results.html', {'votes': votes})


def results(request):
    return JsonResponse({"message": "Vote results will be displayed here!"})

def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def detect_gesture(request):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return HttpResponse("Webcam not accessible", status=500)

    success, frame = cap.read()
    if not success or frame is None:
        return HttpResponse("Failed to read from webcam", status=500)

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # process gestures...

    return HttpResponse("Gesture detected!")
    

def register(request):
    if request.method == 'POST':
        aadhaar = request.POST.get('aadhaar_number')

        # Check if Aadhaar already exists
        if Voter.objects.filter(aadhaar_number=aadhaar).exists():
            return render(request, 'error.html', {
                'error_message': 'Aadhaar number already registered. Please use a different one.'
            })

        try:
            Voter.objects.create(
                full_name=request.POST.get('full_name'),
                father_name=request.POST.get('father_name'),
                age=request.POST.get('age'),
                aadhaar_number=aadhaar,
                house_no=request.POST.get('house_no'),
                street=request.POST.get('street'),
                village_town=request.POST.get('village_town'),
                ward=request.POST.get('ward'),
                mandal=request.POST.get('mandal'),
                district=request.POST.get('district'),
                pincode=request.POST.get('pincode'),
                state=request.POST.get('state'),
                aadhaar_photo=request.FILES.get('aadhaar_photo'),
                face_photo=request.FILES.get('face_photo')
            )
            return redirect('dashboard')  # ✅ Replace with your real dashboard view name

        except IntegrityError:
            return render(request, 'error.html', {
                'error_message': 'Database error occurred while saving voter. Please try again.'
            })

        except Exception as e:
            return render(request, 'error.html', {
                'error_message': f"Unexpected error occurred: {str(e)}"
            })

    return render(request, 'register.html')


def voter_dashboard(request):
    voters = Voter.objects.all()
    return render(request, 'dashboard.html', {'voters': voters})

def check_vote_status(request):
    global vote_submitted
    return JsonResponse({'vote_submitted': vote_submitted})


@csrf_exempt  # Only if CSRF is causing issues, otherwise not needed
def cast_vote(request):
    if request.method == 'POST':
        selected_party_name = request.POST.get('party')
        
        if not selected_party_name:
            return render(request, 'error.html', {'message': 'No party selected.'})

        try:
            party = Party.objects.get(name=selected_party_name)
        except Party.DoesNotExist:
            return render(request, 'error.html', {'message': 'Party not found.'})

        vote_count, created = VoteCount.objects.get_or_create(party=party)
        vote_count.count += 1
        vote_count.save()

        return redirect('thank_you')
    else:
        return render(request, 'error.html', {'message': 'Invalid request method'})


def submit_vote(party_id):
    # blockchain vote submission logic here...

    # update local vote count
    party = Party.objects.get(id=party_id)
    party.vote_count += 1
    party.save()


from django.conf import settings
from django.templatetags.static import static

def get_live_vote_counts(request):
    parties = Party.objects.all()
    results = [
        {
            "name": party.name,
            "votes": party.vote_count,
            "symbol": party.symbol.url if party.symbol and hasattr(party.symbol, 'url') else "/static/default_symbol.png"
        }
        for party in parties
    ]
    print("Live Vote Counts:", results)  # ✅ Check in terminal
    return JsonResponse({"results": results})


def results_page(request):
    return render(request, 'results.html')

def live_results(request):
    return render(request, 'live_results.html')


def verify_face_view(request):
    voter_id = request.session.get('voter_id')
    if not voter_id:
        return redirect('welcome')

    try:
        voter = Voter.objects.get(id=voter_id)
    except Voter.DoesNotExist:
        return redirect('welcome')

    uploaded_img_path = os.path.join(settings.MEDIA_ROOT, str(voter.aadhaar_photo))
    uploaded_image = face_recognition.load_image_file(uploaded_img_path)
    uploaded_encoding = face_recognition.face_encodings(uploaded_image)

    if not uploaded_encoding:
        return render(request, 'verification_failed.html')

    uploaded_encoding = uploaded_encoding[0]

    # Start webcam
    cap = cv2.VideoCapture(0)
    face_verified = False
    live_face_image = None

    for _ in range(50):  # Capture frames and try for some time
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        live_encodings = face_recognition.face_encodings(rgb_frame)

        if live_encodings:
            match = face_recognition.compare_faces([uploaded_encoding], live_encodings[0])
            if match[0]:
                face_verified = True
                live_face_image = frame.copy()
                break

    cap.release()

    if face_verified:
        # Save images to show side-by-side
        live_path = os.path.join(settings.MEDIA_ROOT, f'live_{voter_id}.jpg')
        cv2.imwrite(live_path, live_face_image)

        context = {
            'uploaded_image_url': settings.MEDIA_URL + str(voter.uploaded_photo),
            'live_image_url': settings.MEDIA_URL + f'live_{voter_id}.jpg'
        }
        return render(request, 'face_verified.html', context)
    else:
        return render(request, 'verification_failed.html')


def face_verification(request, voter_id):
    voter = Voter.objects.get(id=voter_id)

    # Load uploaded document image
    doc_image_path = voter.document_photo.path
    doc_image = face_recognition.load_image_file(doc_image_path)
    try:
        doc_encoding = face_recognition.face_encodings(doc_image)[0]
    except IndexError:
        return render(request, "verification_failed.html", {"error": "No face in document."})

    # Start webcam capture
    video = cv2.VideoCapture(0)
    ret, frame = video.read()
    video.release()

    if not ret:
        return render(request, "verification_failed.html", {"error": "Failed to access webcam."})

    # Save the webcam image to temp path for display
    live_path = f'media/temp/live_face_{voter_id}.jpg'
    cv2.imwrite(live_path, frame)

    try:
        live_encoding = face_recognition.face_encodings(frame)[0]
    except IndexError:
        return render(request, "verification_failed.html", {"error": "No face detected in webcam."})

    # Compare faces
    results = face_recognition.compare_faces([doc_encoding], live_encoding)

    if results[0]:
        return render(request, "face_verified.html", {
            "doc_image": voter.document_photo.url,
            "live_image": "/" + live_path  # assuming media is properly served
        })
    else:
        return render(request, "verification_failed.html")


def vote_view(request):
    if request.method == 'POST':
        voter_id = request.POST.get('voter_id')
        voter_name = request.POST.get('voter_name')
        age = request.POST.get('age')
        Party_name = request.POST.get('Party_name')

        # Time tracking
        start_time_str = request.session.get("start_time")
        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str)
            time_taken = (datetime.now() - start_time).total_seconds()
        else:
            time_taken = "N/A"

        filename = 'votes.csv'
        file_exists = os.path.isfile(filename)

        with open(filename, mode='a', newline='') as file:
            fieldnames = ['VoterID', 'VoterName', 'Age', 'Party_name', 'Timestamp', 'TimeTaken']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                'VoterID': voter_id,
                'VoterName': voter_name,
                'Age': age,
                'Party_name': Party_name,
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'TimeTaken': f"{time_taken} seconds"
            })

        return redirect('thank_you1')

    # Set session start time on GET
    request.session['start_time'] = datetime.now().isoformat()
    return render(request, 'vote1.html')


def view_votes(request):
    votes = []
    filename = 'votes.csv'  # Adjust path if stored in a subfolder

    try:
        with open(filename, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                votes.append(row)
    except FileNotFoundError:
        votes = []

    return render(request, 'view_votes.html', {'votes': votes})


def vote_page(request):
    parties = [
        {'name': 'BJP', 'logo': 'bjp.png'},
        {'name': 'Congress', 'logo': 'congress.png'},
        {'name': 'AAP', 'logo': 'aap.png'},
        {'name': 'TMC', 'logo': 'tmc.png'},
        {'name': 'Shiv Sena', 'logo': 'shiv_sena.png'},
        {'name': 'SP', 'logo': 'sp.png'},
        {'name': 'BSP', 'logo': 'bsp.png'},
        {'name': 'CPI', 'logo': 'cpi.png'},
        {'name': 'NOTA', 'logo': 'nota.png'},
    ]

    if request.method == 'POST':
        voter_id = request.POST.get('voter_id')
        voter_name = request.POST.get('voter_name')
        age = request.POST.get('age')
        Party_name = request.POST.get('Party_name')

        # ✅ Combined duplicate check (both ID and name together)
        if Vote.objects.filter(voter_id=voter_id, voter_name__iexact=voter_name).exists():
            messages.error(request, '❌ You have already voted.')
            return redirect('vote1')

        # Save to DB
        Vote.objects.create(voter_id=voter_id, voter_name=voter_name, Party_name=Party_name)

        # Time tracking
        start_time_str = request.session.get("start_time")
        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str)
            time_taken = (datetime.now() - start_time).total_seconds()
        else:
            time_taken = "N/A"

        # Save to CSV
        filename = 'votes.csv'
        file_exists = os.path.isfile(filename)

        with open(filename, 'a', newline='') as file:
            fieldnames = ['VoterID', 'VoterName', 'Age', 'Party_name', 'Timestamp', 'TimeTaken']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                'VoterID': voter_id,
                'VoterName': voter_name,
                'Age': age,
                'Party_name': Party_name,
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'TimeTaken': f"{time_taken} seconds"
            })

        messages.success(request, f'✅ Your vote for {Party_name} has been submitted!')
        return redirect('thank_you1')

    # Set session start time on GET
    request.session['start_time'] = datetime.now().isoformat()
    return render(request, 'vote1.html', {'parties': parties})


def thank_you1(request):
    return render(request, 'thankyou1.html')


PARTY_LIST = ['BJP', 'Congress', 'AAP', 'TMC', 'Shiv Sena', 'SP', 'BSP', 'CPI', 'NOTA']

def results_view(request):
    return render(request, 'live_results.html', {'parties': PARTY_LIST})

def get_vote_data(request):
    vote_counts = {party: 0 for party in PARTY_LIST}
    voters = Voter.objects.exclude(voted_party__isnull=True)

    for voter in voters:
        if voter.voted_party in vote_counts:
            vote_counts[voter.voted_party] += 1

    return JsonResponse(vote_counts)

