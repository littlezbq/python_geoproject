from flask import Flask, render_template, request, jsonify, make_response
from Up_to_Domain.interfaces import up_to_domain
from ascensionpoint_generate.interfaces import interface_visionpoint
from flask_cors import CORS

# app = Flask(__name__, template_folder="static/templates", static_folder="static", static_url_path="")
app = Flask(__name__)
app.config.from_object(__name__)
CORS(app, resources={r'/*': {'origins': '*'}}, supports_credentials=True)
app.jinja_env.variable_start_string = '{['
app.jinja_env.variable_end_string = ']}'


# app.config["SEND_FILE_MAX_AGE_DEFAULT"] = timedelta(seconds=1.0)

@app.route("/", methods=['GET'])
def mainPage():
    return render_template('index.html')


@app.route("/autoSortingPage", methods=['GET'])
def autoSortingPage():
    return render_template('autoSortingPage2.html')


# test
@app.route("/test", methods=['GET', 'POST'])
def test():
    return make_response(jsonify({
        'result': {'msg': 'Success',
                   'code': 0
                   }
    }))


@app.route("/uploadFiles", methods=['GET', 'POST'])
def uploadFiles():
    upload_files = request.files.getlist('files')
    global cur_file_name

    for upload_file in upload_files:
        if upload_file.filename.endswith('.tif'):
            dem_file = upload_file
        else:
            remote_file = upload_file
            cur_file_name = upload_file.filename.split('.')[0]
        upload_file.save('./static/datasets/' + upload_file.filename)

    dem_path = './static/datasets/' + dem_file.filename
    remotePath = './static/datasets/' + remote_file.filename

    path1, path2, centerCoord = up_to_domain.interface_drawRawData(dem_path, remotePath)
    print(centerCoord)
    result = make_response(jsonify({
        'result': {'dem_url': path1,
                   'remote_url': path2,
                   'centerCoord': centerCoord,
                   'code': 0},
    }))
    return result


@app.route("/distantCalc", methods=['GET', 'POST'])
def distantCalc():
    # global cur_file_name
    global timelimit
    global demAxis_use

    dis = request.json['time_limit']
    demAxis = request.json['demAxis']

    dis = int(dis)
    demAxis = float(demAxis)

    timelimit = dis
    demAxis_use = demAxis

    dem_path = './static/datasets/' + cur_file_name + '.tif'
    remotePath = './static/datasets/' + cur_file_name + '.png'

    centerpoint, accessdomain_path, location = up_to_domain.interface_uptodomain(dem_path, remotePath, dis, demAxis)

    print(accessdomain_path,location)

    # if isinstance(accesspoints,np.ndarray):
    #     accesspointsUse = accesspoints.tolist()
    # else:
    #     accesspointsUse = accesspoints

    # Only get the accessdomain points in format longtitude/latitude
    result = make_response(jsonify({
        'result': {'accessdomain_path': accessdomain_path,
                   'centerpoint': centerpoint,
                   'location': location,
                   'code': 0},
    }))
    return result


@app.route("/calAscensionPoint", methods=['GET', 'POST'])
def calAscensionPoint():
    demPath = './static/datasets/' + cur_file_name + '.tif'

    demAxis = demAxis_use

    ascensionPointNum = int(request.json['ascensionPointNum'])
    divideSpace = float(request.json['divideSpace'])

    path1, path2 = interface_visionpoint.interface_cal_ascensionpoint(demPath, ascensionPointNum, timelimit, divideSpace, demAxis)

    result = make_response(jsonify({
        'result': {'ascension_point_list': path1,
                   'ascensionpoint_viewshedpathlist': path2,
                   'code': 0},
    }))

    return result


@app.route("/calExposivePoint", methods=['GET', 'POST'])
def calExposivePoint():
    global cur_file_name
    global timelimit

    dem_path = './static/datasets/' + cur_file_name + '.tif'
    exposivePointNum = int(request.json['exposivePointNum'])

    exposivepoint_numlist_absolute, exposive_result_path = interface_visionpoint.interface_cal_exposivepoint(dem_path, timelimit, exposivePointNum)

    result = make_response(jsonify({
        'result': {'exposivepoint_numlist_absolute': exposivepoint_numlist_absolute,
                   'exposive_result_path': exposive_result_path,
                   'code': 0},
    }))

    return result


# wy special
@app.route("/wyFunction1", methods=["GET", "POST"])
def wyFunction1():
    return False


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(port=8000)

    # from commonutils.tools import  Tools
    # tl = Tools()
    #
    # tl.clear_cache()
    # tl.clear_result(0)
