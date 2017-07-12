const Buffer = require("buffer").Buffer;
const cv = require("opencv");
const DSCClient = require("dscframework/clients/node");
const DSCUtil = require("../dscframework/Util");

class FaceDetectionSensor {

  constructor() {
    this.dsc = new DSCClient("ws://localhost:8080");
    this.running = false;
    this.cvRunning = false;
    this.positions = [];
  }

  start() {
    this.dsc.start(()=>this.onStart());
  }

  onStart() {
    this.running = true;
    this.dsc.register("facedetect", {
      input: {},
      output: {}
    });

    this.dsc.subscribe("camera", (head, data) => this.onCameraData(head, data));
  }

  onCameraData(head, data) {
    this.cvRunning = true;
    var b64 = DSCUtil.uint8ToString(data);
    var buffer = Buffer.from(b64.substring(b64.indexOf(",") + 1), "base64");
    cv.readImage(buffer, (err, matrix) => {
      if (err) {
        console.log(err);
        this.cvRunning = false;
      } else {
        matrix.detectObject(
          cv.FACE_CASCADE,
          {},
          (err, data) => {
            if (err) {
              console.log(err, data);
            } else {
              this.onDetection(head, matrix, data);
            }
            this.cvRunning = false;
          }
        );
      }
    });
  }

  onDetection(head, matrix, data) {
    if ( !this.hasUpdate(data) ) return;
    var n = data.length, i = 0, x = 0, y = 0;
    var rows = 112;
    var columns = 92;
    var size = rows * columns;
    var nmat = null, loc = null, row = null;
    var buf = new Uint8Array(n * size)
    var matgs = matrix.greyscale();

    for (; i < n; i++) {
      loc = data[i];
      cmat = matgs.crop(loc.x, loc.y, loc.width, loc.height).resize(rows, columns);
      for (y = 0; y < rows; y++) { // TODO, resize and extract buffer, or just handle everything here?
        for (x = 0; x < columns; x++) {
          val = cmat.get(x,y) / 255.0;
          buf[x + y * columns + i * size] = val;
        }
      }
    }

    //TODO remove this broadcast, and turn it into the buffer broadcast
    this.dsc.broadcast("facedetect", head, data);
    this.dsc.broadcast("facebuffer", head, buf);
  }

  hasUpdate(data) {
    var len = data.length, i = 0, buf = [], update = false;
    if (len !== this.positions.length) return true;

    for(; i < len; i++) {
      buf.push(this.createPosition(data[i]));
      if ( this.distance(buf[i], this.positions[i]) > 1.0 ) update = true;
    }

    this.positions = buf;
    return update;
  }

  createPosition(loc) {
    // TODO base Z off average theta trig
    var z = Math.sqrt(loc.width * loc.width + loc.height * loc.height);
    return [loc.x, loc.y, z];
  }

  distance(a, b) { // TODO trig
    var dz = a[2] - b[2];
    var dx = (a[0] - b[0]) * dz;
    var dy = (a[1] - b[1]) * dz;
    return Math.sqrt(dx*dx + dy*dy + dz*dz);
  }

}

var sensor = new FaceDetectionSensor();
sensor.start();
