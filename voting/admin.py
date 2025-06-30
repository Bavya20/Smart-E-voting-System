from django.contrib import admin
from .models import Voter, Vote, BlockchainRecord


@admin.register(Voter)
class VoterAdmin(admin.ModelAdmin):
    list_display = ('id', 'full_name', 'has_voted', 'aadhaar_number', 'age', 'district', 'state')
    list_filter = ('has_voted', 'id')
    search_fields = ('full_name', 'aadhaar_number')
    readonly_fields = ('registered_on',)
    ordering = ('-registered_on',)


@admin.register(Vote)
class VoteAdmin(admin.ModelAdmin):
    list_display = ('voter_id', 'voter_name', 'age', 'Party_name')
    list_filter = ('voter_id', 'Party_name')
    search_fields = ('voter__full_name', 'party')


@admin.register(BlockchainRecord)
class BlockchainRecordAdmin(admin.ModelAdmin):
    list_display = ('block_index', 'block_hash', 'previous_hash', 'timestamp')
    search_fields = ('block_hash', 'previous_hash')
    ordering = ('-block_index',)