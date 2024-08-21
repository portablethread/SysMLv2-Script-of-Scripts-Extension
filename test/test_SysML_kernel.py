#!/usr/bin/env python3
#
# This file is part of Script of Scripts (SoS), a workflow system
# for the execution of commands and scripts in different languages.
# Please visit https://github.com/vatlab/SOS for more information.
#
# Copyright (C) 2016 Bo Peng (bpeng@mdanderson.org)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

#
# NOTE: for some namespace reason, this test can only be tested using
# nose.
#
# % nosetests test_kernel.py
#
#
import os
import unittest

from ipykernel.tests.utils import execute, wait_for_idle
from sos_notebook.test_utils import (clear_channels, get_display_data,
                                     get_result, get_std_output, sos_kernel)


class TestSysMLKernel(unittest.TestCase):
    #
    # Beacuse these tests would be called from sos/test, we
    # should switch to this directory so that some location
    # dependent tests could run successfully
    #
    def setUp(self):
        self.olddir = os.getcwd()
        if os.path.dirname(__file__):
            os.chdir(os.path.dirname(__file__))

    def tearDown(self):
        os.chdir(self.olddir)

    def testGetPythonDataFrameFromSysML(self):
        # Python -> SysML
        with sos_kernel() as kc:
            iopub = kc.iopub_channel
            # create a data frame
            execute(kc=kc, code='''
import pandas as pd
import numpy as np
arr = np.random.randn(1000)
arr[::10] = np.nan
df = pd.DataFrame({'column_{0}'.format(i): arr for i in range(10)})
''')
            clear_channels(iopub)
            execute(kc=kc, code="%use SysML")
            _, stderr = get_std_output(iopub)
            self.assertEqual(stderr, '', "GOT ERROR {}".format(stderr))
            execute(kc=kc, code="%get df")
            wait_for_idle(kc)
            execute(kc=kc, code="df.size()")
            res = get_display_data(iopub)
            self.assertEqual(res, '1000')
            execute(kc=kc, code="df.vectors().to_a().size()")
            res = get_display_data(iopub)
            self.assertEqual(res, '10')
            execute(kc=kc, code="%use sos")
            wait_for_idle(kc)
        
    def testGetPythonMatrixFromSysML(self):
        # Python -> SysML
        with sos_kernel() as kc:
            iopub = kc.iopub_channel
            # create a matrix 
            execute(kc=kc, code='''
import numpy as np
mat_var = np.matrix([[1,2],[3,4]])
''')
            clear_channels(iopub)
            execute(kc=kc, code="%use SysML")
            wait_for_idle(kc)
            execute(kc=kc, code="%get mat_var")
            wait_for_idle(kc)
            execute(kc=kc, code="mat_var.size()")
            res = get_display_data(iopub)
            self.assertEqual(res, '4')
            execute(kc=kc, code="%use sos")
            wait_for_idle(kc)
            #

    def testGetPythonNoneFromSysML(self):
        # Python -> SysML
        with sos_kernel() as kc:
            iopub = kc.iopub_channel
            # create a data frame
            execute(kc=kc, code='''
null_var = None
''')
            clear_channels(iopub)
            execute(kc=kc, code="%use SysML")
            wait_for_idle(kc)
            execute(kc=kc, code="%get null_var")
            wait_for_idle(kc)
            execute(kc=kc, code="null_var == nil")
            res = get_display_data(iopub)
            self.assertEqual(res, 'true')
            execute(kc=kc, code="%use sos")
            wait_for_idle(kc)
            #

    def testGetPythonDataFromSysML(self):
        with sos_kernel() as kc:
            iopub = kc.iopub_channel
            execute(kc=kc, code='''
null_var = None
num_var = 123
import numpy
import pandas
num_arr_var = numpy.array([1, 2, 3])
logic_var = True
logic_arr_var = [True, False, True]
char_var = '1"23'
char_arr_var = ['1', '2', '3']
list_var = [1, 2, '3']
dict_var = dict(a=1, b=2, c='3')
set_var = {1, 2, '3'}
mat_var = numpy.matrix([[1,2],[3,4]])
recursive_var = {'a': {'b': 123}, 'c': True}
comp_var = 1+2j
''')
            wait_for_idle(kc)
            execute(kc=kc, code='''
%use SysML
%get null_var num_var num_arr_var logic_var logic_arr_var char_var char_arr_var set_var list_var dict_var mat_var recursive_var comp_var
%dict -r
%put null_var num_var num_arr_var logic_var logic_arr_var char_var char_arr_var set_var list_var dict_var mat_var recursive_var comp_var
%use sos
''')
            wait_for_idle(kc)
            execute(kc=kc, code='''
%dict null_var num_var num_arr_var logic_var logic_arr_var char_var char_arr_var set_var list_var dict_var mat_var recursive_var comp_var
''')
            res = get_result(iopub)
            self.assertEqual(res['null_var'], None)
            self.assertEqual(res['num_var'], 123)
            self.assertEqual(res['num_arr_var'], [1,2,3])
            self.assertEqual(res['logic_var'], True)
            self.assertEqual(res['logic_arr_var'], [True, False, True])
            self.assertEqual(res['char_var'], '1"23')
            self.assertEqual(res['char_arr_var'], ['1', '2', '3'])
            self.assertEqual(res['set_var'], {1, 2, '3'})
            self.assertEqual(res['list_var'], [1,2,'3'])
            self.assertEqual(res['dict_var'], {'a': 1, 'b': 2, 'c': '3'})
            self.assertEqual(res['mat_var'].shape, (2, 2))
            self.assertEqual(res['recursive_var'],  {'a': {'b': 123}, 'c': True})
            self.assertEqual(res['comp_var'], (1+2j))

#dataframe

    def testPutSysMLDataToPython(self):
        with sos_kernel() as kc:
            iopub = kc.iopub_channel
            # create a data frame
            execute(kc=kc, code="""\
%use SysML
null_var = nil
num_var = 123
num_arr_var = [1, 2, 3]
logic_var = true
logic_arr_var = [true, true, false]
char_arr_var = [1, 2, "3"]
char_var = '1"23'
mat_var = N[[1, 2], [3, 4]]
recursive_var = {"a"=>1, "b"=>{"c"=>3}, "d"=>"whatever"}
comp_var = Complex(1,2)
single_char_var = 'a'
""")
            wait_for_idle(kc)
            execute(kc=kc, code="%put null_var num_var num_arr_var logic_var logic_arr_var char_arr_var char_var mat_var recursive_var comp_var single_char_var")
            wait_for_idle(kc)
            execute(kc=kc, code="%use sos")
#            wait_for_idle(kc)
#            execute(kc=kc, code='''
#%use sos
#named_list_var = list(named_list_var)
#''')
            wait_for_idle(kc)
            execute(kc=kc, code="%dict null_var num_var num_arr_var logic_var logic_arr_var char_arr_var char_var mat_var recursive_var comp_var single_char_var")
            res = get_result(iopub)
            self.assertEqual(res['null_var'], None)
            self.assertEqual(res['num_var'], 123)
            self.assertEqual(list(res['num_arr_var']), [1,2,3])
            self.assertEqual(res['logic_var'], True)
            self.assertEqual(res['logic_arr_var'], [True, True, False])
            self.assertEqual(res['char_var'], '1"23')
            self.assertEqual(res['char_arr_var'], [1, 2, '3'])
            self.assertEqual(res['mat_var'].shape, (2,2))
            self.assertEqual(res['recursive_var'], {'a': 1, 'b': {'c': 3}, 'd': 'whatever'})
            self.assertEqual(res['comp_var'], 1+2j)
            self.assertEqual(res['single_char_var'], 'a')
            execute(kc=kc, code="%use sos")
            wait_for_idle(kc)


if __name__ == '__main__':
    unittest.main()
