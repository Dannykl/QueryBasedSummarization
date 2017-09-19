from django.shortcuts import render
from django.http import HttpResponseRedirect

from .models import Entry
from django.utils import timezone 
from .forms import NameForm
from .Summarization import *

def home(request):
    return render(request, 'blog/home.html', {})
  

def blog(request):
	entries = Entry.objects.filter(published_date__lte=timezone.now()).order_by('-published_date')
	return render(request, 'blog/blog.html', {'posts':entries})

# def experience(request):
# 	return render(request, 'blog/experience.html',{})
def experience(request):
	form = NameForm()
	if(request.method == 'POST'):
		form = NameForm(request.POST)     
		
		if form.is_valid():
			query = form.cleaned_data['query']
			summary = Summarization.main(query)
			print(summary)
			return render(request, 'blog/experience.html',{'summary': summary})
			#return HttpResponseRedirect('',summary)
	else:
		form = NameForm()

	return render(request, 'blog/experience.html', {'form'   : form})
