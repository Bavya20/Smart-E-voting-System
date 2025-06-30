function verifyVoter() {
    let formData = new FormData();
    formData.append("aadhaar_number", document.getElementById("aadhaar_number").value);
    formData.append("image", document.getElementById("face_image").files[0]);

    fetch('/verify_voter/', { method: 'POST', body: formData })
        .then(response => response.json())
        .then(data => document.getElementById("message").innerText = data.message);
}

function castVote() {
    fetch('/cast_vote/', { method: 'POST' })
        .then(response => response.json())
        .then(data => document.getElementById("message").innerText = data.message);
}
