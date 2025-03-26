from django import forms
from .models import Participant

class ParticipantAdminForm(forms.ModelForm):
    height = forms.FloatField(label="Height", help_text="Enter height in centimeters", required=True)
    class Meta:
        model = Participant
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields['height'].widget.attrs.update({'step': '0.1'})
        self.fields['height'].help_text += """
            <div style="margin-top: 0.5em;">
              <label><input type="checkbox" id="toggle-imperial" /> Input in feet & inches</label>
              <div id="imperial-fields" style="display: none; margin-top: 0.5em;">
                <input type="number" id="feet-input" placeholder="Feet" style="width: 80px;" />
                <input type="number" id="inches-input" placeholder="Inches" style="width: 80px;" />
              </div>
            </div>
        """

    class Media:
        js = ('admin/js/height_toggle.js',)