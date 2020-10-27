# import external libraries.
import os, csv, statistics

# define notable locations.
DATA_DIR = r'./datacollection_ck'
COMPILED_FILE = r'./compiled.csv'
SUMMARY_FILE = r'./aggregate.csv'

# all the headers from the data sets.
HEADERS = set()
additional_headers = ['project', 'release', 'maven_reuse', 'maven_release', 'class_count']
removed_headers = ['file', 'class', 'type']

# placeholder list of lines to write to compiled.csv
compiled_dict = dict()
summary_dict = dict()

# get all folders in DATA_DIR.
# https://stackoverflow.com/questions/7781545/how-to-get-all-folder-only-in-a-given-path-in-python
project_folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

# for each folder in the DATA_DIR, open the class.csv file and copy content into
# the compiled csv file.
for project in project_folders:
    # create an empty dictonary for this project in the compiled_dict and
    # summary_dict.
    compiled_dict[project] = dict()
    summary_dict[project] = dict()

    # define shortcut variable.
    project_folder = os.path.join(DATA_DIR, project)

    # find all release folders in the project_folder.
    release_folders = [d for d in os.listdir(project_folder) if os.path.isdir(os.path.join(project_folder, d))]

    # for each release_folder extract the contents of class.csv to compile.csv.
    for release in release_folders:
        print('compiling: '+project+': '+release)

        # create an empty dictonary for this release in the compiled_dict and
        # summary_dict.
        compiled_dict[project][release] = dict()
        summary_dict[project][release] = dict()

        # initialise the class count for this release.
        summary_dict[project][release]['class_count'] = 0

        # define shortcut variable.
        release_folder = os.path.join(project_folder, release)

        # get all class.csv files (should only be one).
        for file in [f for f in os.listdir(release_folder) if f == 'class.csv']:
            # catch any errors: eg, no rows in csv file.
            try:
                # open the csv file to read all rows.
                with open(os.path.join(release_folder, file), 'r') as f_read:
                    reader = csv.reader(f_read)

                    # get the header row.
                    header = next(reader)

                    # ensure all the elements of this header is in the HEADERS list.
                    for h in header:
                        HEADERS.add(h)

                    # add each data row to the compiled_list. do not copy header row.
                    for row in [r for r in reader if len(r) > 1 and r[2] == 'class']:
                        # r[1] == class name. create a new dict for this class.
                        compiled_dict[project][release][row[1]] = dict()

                        # increment the class_count for this release.
                        summary_dict[project][release]['class_count'] += 1

                        # iterate over each column and create a key value pair based on the header
                        # and value at each column.
                        for i in range(0, len(row)):
                            compiled_dict[project][release][row[1]][header[i]] = row[i]

            # if there is an error with the csv, continue to next project.
            except: continue

    # check if there is a maven_reuse csv file in the project directory.
    for maven_reuse_file_name in [f for f in os.listdir(project_folder) if 'maven_reuse.csv' in f]:
        # define shortcut variable.
        maven_reuse_file_path = os.path.join(project_folder, maven_reuse_file_name)

        # open the maven_reuse file, if there is a release in that file that we
        # have metric for, update the reuse amount and release date values.
        with open(maven_reuse_file_path, 'r') as f_read:
            reader = csv.reader(f_read)

            # iterate over each row and check if there is a valid release.
            for row in reader:
                # check if this release is in our compiled dictionary.
                if row[0] in summary_dict[project]:
                    # if the release is in our compiled dictionary, update the
                    # reuse amount and release date.
                    summary_dict[project][row[0]]['maven_reuse'] = row[1]
                    summary_dict[project][row[0]]['maven_release'] = row[2]

# write all class values to the compiled file.
with open(COMPILED_FILE, 'w', newline='') as f_write:
    writer = csv.writer(f_write)

    # write a line of headers including the new columns.
    writer.writerow(additional_headers + list(HEADERS))

    # write each class entry into the compiled_csv.
    for project in compiled_dict:
        for release in compiled_dict[project]:
            for class_name in compiled_dict[project][release]:
                # shortcut variable.
                c = compiled_dict[project][release][class_name]

                # write this class information into the COMPILED_FILE.
                writer.writerow([project, release] + [c[h] if h in c and c[h] != '' else 'NaN' for h in list(HEADERS)])

# summarise the information.
for project in compiled_dict:
    # for each release in this project, summarise the classes in this release.
    for release in compiled_dict[project]:
        print('summarizing: '+project+': '+release)

        # for each column that is a metric add aggregate metrics.
        for metric in [m for m in list(HEADERS) if m not in removed_headers]:
            summary_dict[project][release][metric] = {
                'count': 0,
                'sum': 0,
                'average': 0,
                'median': 0,
                'stdev': 0,
                'min': 0,
                'max': 0,
                'list': [],
            }

        # iterate over each class in the release and
        for class_name in compiled_dict[project][release]:

            # iterate over each metric in this class.
            for metric in [m for m in compiled_dict[project][release][class_name] if m not in removed_headers and compiled_dict[project][release][class_name][m] != 'NaN' and compiled_dict[project][release][class_name][m] != '']:
                # shortcut value.
                value = float(compiled_dict[project][release][class_name][metric])

                # increment sum by the metric value.
                summary_dict[project][release][metric]['sum'] += float(compiled_dict[project][release][class_name][metric])

                # check if this is the min value.
                if value < summary_dict[project][release][metric]['min']:
                    summary_dict[project][release][metric]['min'] = value

                # check if this is the max value.
                if value > summary_dict[project][release][metric]['max']:
                    summary_dict[project][release][metric]['max'] = value

                # add this value to the list so we can calculate median.
                summary_dict[project][release][metric]['list'].append(value)

                # increment count by 1.
                summary_dict[project][release][metric]['count'] += 1

        # calculate the average value.
        # for each column that is a metric add aggregate metrics.
        for metric in [m for m in list(HEADERS) if m not in removed_headers]:
            # if there is a count of zero, enter 'NaN'.
            if summary_dict[project][release][metric]['count'] == 0:
                summary_dict[project][release][metric]['sum'] = 'NaN'
                summary_dict[project][release][metric]['average'] = 'NaN'
                summary_dict[project][release][metric]['stdev'] = 'NaN'
                summary_dict[project][release][metric]['median'] = 'NaN'
                summary_dict[project][release][metric]['min'] = 'NaN'
                summary_dict[project][release][metric]['max'] = 'NaN'

            # if there is a count greater than zero, calculate average and median.
            else:
                # calculate average.
                summary_dict[project][release][metric]['average'] = statistics.mean(summary_dict[project][release][metric]['list'])

                # calculate median.
                summary_dict[project][release][metric]['median'] = statistics.median(summary_dict[project][release][metric]['list'])

                # calculate standard deviation. catch error if there is only 1
                # data observation.
                try: summary_dict[project][release][metric]['stdev'] = statistics.stdev(summary_dict[project][release][metric]['list'])
                except: summary_dict[project][release][metric]['average'] = 'NaN'

# write all the summary values to the summary file.
with open(SUMMARY_FILE, 'w', newline='') as f_write:
    writer = csv.writer(f_write)

    # create a list of summary headers.
    summary_headers = additional_headers

    # for each metric, add extra summary headers.
    for header in [h for h in HEADERS if h not in removed_headers]:
        summary_headers.append(header+'_sum')
        summary_headers.append(header+'_average')
        summary_headers.append(header+'_stdev')
        summary_headers.append(header+'_median')
        summary_headers.append(header+'_min')
        summary_headers.append(header+'_max')

    # write a line of headers including the new columns.
    writer.writerow(summary_headers)

    # write each class entry into the compiled_csv.
    for project in summary_dict:
        for release in summary_dict[project]:
            # create a new row of values for this release.
            row = ['']*len(summary_headers)

            # input project and release values.
            row[0] = project
            row[1] = release
            row[2] = summary_dict[project][release]['maven_reuse'] if 'maven_reuse' in summary_dict[project][release] else ''
            row[3] = summary_dict[project][release]['maven_release'] if 'maven_release' in summary_dict[project][release] else ''
            row[4] = summary_dict[project][release]['class_count']

            # for each metric that this project has recorded, add each aggreate
            # value to the summary csv file.
            for metric in [m for m in summary_dict[project][release] if m not in additional_headers]:
                # for each aggregate item for this metric, add it as a column
                # to the row.
                for item in ['sum', 'average', 'stdev', 'median', 'min', 'max']:
                    # determine which column index this item refers to.
                    index = summary_headers.index(metric+'_'+item)

                    # add this value to the row.
                    row[index] = summary_dict[project][release][metric][item]

            # write this class information into the SUMMARY_FILE.
            writer.writerow(row)
