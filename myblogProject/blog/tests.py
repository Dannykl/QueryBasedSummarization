from django.test import TestCase
from .models import Entry
# Create your tests here.
class EntryModelTest(TestCase):

    def test_string_representation(self):
        entry = Entry(title="my entry title")
        self.assertEqual(str(entry), entry.title)
    def test_verbose_name_plural(self):
    	self.assertEqual(str(Entry._meta.verbose_name_plural),"entries")

