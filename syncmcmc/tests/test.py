
from unittest import TestCase

from syncmcmc import testprog as tp

class TestMe(TestCase):
    def i_am_string(self):
        s = tp.myfunc()
        self.assertTrue(isinstance(s,basestring))