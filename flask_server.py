from flask import Flask, render_template, url_for
import main
app = Flask('Map', static_url_path='/static')


@app.route('/')
def index():
    main.getmap()
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(debug=True)