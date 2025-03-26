from django.contrib import admin
from .models import Participant
from .forms import ParticipantAdminForm
import re

# Register your models here.
class ParticipantAdmin(admin.ModelAdmin):
    form = ParticipantAdminForm

    def get_changeform_initial_data(self, request):
        initial = super().get_changeform_initial_data(request)

        pattern = re.compile(r'^P(\d{3})$')
        valid_ids = Participant.objects.filter(identifier__regex=r'^P\d{3}$')

        # Find the next available identifier
        max_num = 0
        for participant in valid_ids:
            match = pattern.match(participant.identifier)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)

        next_id = f'P{max_num + 1:03d}'
        initial['identifier'] = next_id

        return initial

admin.site.register(Participant, ParticipantAdmin)
