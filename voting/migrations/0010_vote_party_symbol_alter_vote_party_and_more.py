# Generated by Django 5.1.1 on 2025-04-18 09:29

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('voting', '0009_alter_voter_is_disabled_alter_voter_vote_type'),
    ]

    operations = [
        migrations.AddField(
            model_name='vote',
            name='party_symbol',
            field=models.ImageField(blank=True, null=True, upload_to='vote_symbols/'),
        ),
        migrations.AlterField(
            model_name='vote',
            name='party',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='voting.party', verbose_name='Selected Party'),
        ),
        migrations.AlterField(
            model_name='voter',
            name='voted_party',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='voting.party'),
        ),
    ]
