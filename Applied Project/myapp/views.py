from django.shortcuts import redirect, render
from .models import Document
from .forms import DocumentForm
import os

def my_view(request):
    print(f"Great! You're using Python 3.6+. If you fail here, use the right version.")
    message = 'Upload a wav file to analyse!'




    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()
            os.system('python3 /Users/kian/Workspace/applied_2/speech/myapp/main.py')
            print("!!!!!The file has been uploaded!!!!!!")
            # Redirect to the document list after POST  
            path = "/Users/kian/Workspace/applied_2/speech/myapp/models/speaker_recognition/16000_pcm_speeches/audio/random/"
            for i in os.listdir(path):
                if i.endswith(".wav"):
                    os.remove(os.path.join(path,i))   

            return redirect('my-view')
    
        else:
            print("-----Error-----")
            message = 'The form is not valid. Fix the following error:'
    else:
        form = DocumentForm()  # An empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()

    # Render list page with the documents and the form
    context = {'documents': documents, 'form': form, 'message': message}
    return render(request, 'list.html', context)


