#coding:utf-8
from flask import Flask, request, jsonify
from flask_restful import Api,Resource,reqparse
from serve import get_model_api


# define the app
app = Flask(__name__)

# 先绑定一个api，进行初始化操作
api = Api(app)
model_api = get_model_api()
class NerView(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        # reqparse 是一个类似于WTforms验证的一个模板，用这个模板的时候，需要先进行引用，然后和WTForms的功能就差不了，就是一个验证用户输入的功能。
        parser.add_argument('input_conll' ,type = str , help = 'please input the conll string for entity recognition!' ,required = True)
        # 定义一个username,说明用户需要传入关于username的一个值()后面的都是参数.括号里面的参数可以先不考虑
        args = parser.parse_args()
        # 对用户传入的参数进行解析，不解析的话，是会报错的
        input_conll = args['input_conll']

        output_conll = model_api(input_conll)
        return {"output_conll" : output_conll}

api.add_resource(NerView , '/')
if __name__ == '__main__':
    app.run()