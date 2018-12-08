"""
Copyright(c), Google, LLC (Andrew Ferlitsch)
Copyright(c), Virtualdvid (David Molina)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# number_verify: Checks each tests funtions names and
# re-enumerate if names are duplicated.


import glob

test_files = [f for f in glob.glob("*_test.py")]

for file in test_files:
    f = open(file, 'r')
    func_list = [line[8:16] for line in f if line[8:13] == 'test_']
    func_unic = []
    func_dupl = []
    for func in func_list:
        if func not in func_unic:
            func_unic.append(func)
        else:
            func_dupl.append(func)

    if not func_dupl:
        print('Checked module {} not duplicate functions were found'.format(file))
    else:
        f = open(file, 'r')
        f_name = file.split('.')
        new_file_name = f_name[0] + '_cp.py'
        f2 = open(new_file_name, 'w+')
        i = 1
        for line in f:
            if 'test_' in line and 'x' not in line:
                f2.write(line.replace(line[8:16], 'test_' + str(i).zfill(3)))
                i += 1
            else:
                f2.write(line)
        print('Checked module {} this functions were duplicates {} and fixed on {}'.format(file, func_dupl, new_file_name))
        f2.close()
    f.close()
