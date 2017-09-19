from django import forms


class NameForm(forms.Form):
    query = forms.CharField(widget=forms.TextInput(attrs={'size': '140','placeholder': 'Enter your query regarding about BP Oil Disaster. eg. how many people were killed in BP oil disaster?'}),label='Query ', required = True,max_length=1000)

