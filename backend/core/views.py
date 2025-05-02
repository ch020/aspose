import os

from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Participant

from .upload_handlers import PrintProgressUploadHandler


# Login View
@api_view(['POST'])
def login(request):
    identifier = request.data.get('identifier')
    try:
        participant = Participant.objects.get(identifier=identifier)
        return Response({
            "success": True,
            "identifier": participant.identifier,
            "height": participant.height,
        })
    except Participant.DoesNotExist:
        return Response({"success": False}, status=404)

# Upload Zip Handler
@api_view(['POST'])
def upload_zip(request):
    request.upload_handlers.insert(0, PrintProgressUploadHandler(request))

    file = request.FILES.get('file')
    if not file:
        return Response({"success": False, "error": "No file provided"}, status=400)

    save_path = os.path.join(settings.ZIP_SAVE_DIRECTORY, file.name)
    with open(save_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    return Response({'success': True, 'filename': file.name})
