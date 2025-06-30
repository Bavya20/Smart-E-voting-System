from django.db import models

class Party(models.Model):
    name = models.CharField(max_length=100, unique=True)
    symbol = models.ImageField(upload_to='symbols/', blank=True, null=True)

    def __str__(self):
        return self.name

class Voter(models.Model):
    full_name = models.CharField(max_length=100, default='Unknown')
    father_name = models.CharField(max_length=100, default='Unknown')
    age = models.IntegerField(default=18)
    aadhaar_number = models.CharField(max_length=12, unique=True)
    house_no = models.CharField(max_length=50, default='N/A')
    street = models.CharField(max_length=100, default='N/A')
    village_town = models.CharField(max_length=100, default='N/A')
    ward = models.CharField(max_length=10, default='N/A')
    mandal = models.CharField(max_length=100, default='N/A')
    district = models.CharField(max_length=100, default='N/A')
    pincode = models.CharField(max_length=10, default='000000')
    state = models.CharField(max_length=100, default='N/A')
    aadhaar_photo = models.ImageField(upload_to='aadhaar_photos/')
    face_photo = models.ImageField(upload_to='face_photos/')
    registered_on = models.DateTimeField(auto_now_add=True)
    has_voted = models.BooleanField(default=False)
    #voted_party = models.ForeignKey('Party', on_delete=models.SET_NULL, null=True, blank=True)
    voted_party = models.CharField(max_length=12, default='Unknown')
    aadhaar_photo = models.ImageField(upload_to='aadhaar_photos/')

    def __str__(self):
        return f"{self.full_name} ({self.aadhaar_number})"

class Vote(models.Model):
    voter_id = models.CharField(max_length=100, unique=True)
    voter_name = models.CharField(max_length=100, default='Unknown')
    age = models.IntegerField(default=18)
    Party_name = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.voter_id} - {self.Party_name}"

class BlockchainRecord(models.Model):
    """Stores blockchain records for vote integrity and security"""
    block_index = models.IntegerField(unique=True, verbose_name="Block Index")
    block_hash = models.CharField(max_length=64, unique=True, verbose_name="Block Hash")
    previous_hash = models.CharField(max_length=64, verbose_name="Previous Block Hash")
    timestamp = models.DateTimeField(auto_now_add=True, verbose_name="Block Timestamp")
    votes = models.JSONField(verbose_name="Votes Data")

    def __str__(self):
        return f"Block {self.block_index} - {self.block_hash[:10]}..."
