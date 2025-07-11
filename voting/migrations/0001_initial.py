# Generated by Django 5.1.1 on 2025-04-09 10:05

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='BlockchainRecord',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('block_index', models.IntegerField(unique=True, verbose_name='Block Index')),
                ('block_hash', models.CharField(max_length=64, unique=True, verbose_name='Block Hash')),
                ('previous_hash', models.CharField(max_length=64, verbose_name='Previous Block Hash')),
                ('timestamp', models.DateTimeField(auto_now_add=True, verbose_name='Block Timestamp')),
                ('votes', models.JSONField(verbose_name='Votes Data')),
            ],
        ),
        migrations.CreateModel(
            name='Party',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Voter',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('full_name', models.CharField(default='Unknown', max_length=100)),
                ('father_name', models.CharField(default='Unknown', max_length=100)),
                ('age', models.IntegerField(default=18)),
                ('aadhaar_number', models.CharField(max_length=12, unique=True)),
                ('house_no', models.CharField(default='N/A', max_length=50)),
                ('street', models.CharField(default='N/A', max_length=100)),
                ('village_town', models.CharField(default='N/A', max_length=100)),
                ('ward', models.CharField(default='N/A', max_length=10)),
                ('mandal', models.CharField(default='N/A', max_length=100)),
                ('district', models.CharField(default='N/A', max_length=100)),
                ('pincode', models.CharField(default='000000', max_length=10)),
                ('state', models.CharField(default='N/A', max_length=100)),
                ('aadhaar_photo', models.ImageField(upload_to='aadhaar_photos/')),
                ('face_photo', models.ImageField(upload_to='face_photos/')),
                ('registered_on', models.DateTimeField(auto_now_add=True)),
                ('has_voted', models.BooleanField(default=False)),
            ],
        ),
        migrations.CreateModel(
            name='VoteCount',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('count', models.IntegerField(default=0)),
                ('party', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='voting.party')),
            ],
        ),
        migrations.CreateModel(
            name='Vote',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('party', models.CharField(max_length=50, verbose_name='Selected Party')),
                ('timestamp', models.DateTimeField(auto_now_add=True, verbose_name='Vote Timestamp')),
                ('voter', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='voting.voter', verbose_name='Voter')),
            ],
        ),
    ]
