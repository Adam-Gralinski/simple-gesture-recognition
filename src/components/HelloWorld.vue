<template>
  <div class="hello">
    <h1>{{ msg }}</h1>
    <video class="video" ref="video" width="500" height="500"></video>
    <canvas ref="output"/>
    <button @click="armStartSnapshot">Set starting pose</button>
  </div>
</template>

<script>
/*
  This code needs refactor, please don't jugde me by that.

  It was one friday PoC project to show how easy is to use TensorflowJS, 
  and imperative approach to change the model from recognizing body parts, 
  to recognize simple gesture.
*/

/* eslint-disable */
import * as posenet from "@tensorflow-models/posenet";
import {drawBoundingBox, drawKeypoints, drawSkeleton} from './utils';

export default {
  name: "HelloWorld",
  props: {
    msg: String
  },
  data() {
    return {
      imageScaleFactor: 0.5,
      flipHorizontal: false,
      outputStride: 16,
      videoWidth: 500,
      armInitialPosition: null,
      last50Poses: [],
      poses: [],
      goDownCounter: 0,
      recentlySended: false,
      guiState: {
        algorithm: 'single-pose',
        input: {
          mobileNetArchitecture: '0.50',
          outputStride: 16,
          imageScaleFactor: 0.5,
        },
        singlePoseDetection: {
          minPoseConfidence: 0.1,
          minPartConfidence: 0.5,
        },
        multiPoseDetection: {
          maxPoseDetections: 5,
          minPoseConfidence: 0.15,
          minPartConfidence: 0.1,
          nmsRadius: 30.0,
        },
        output: {
          showVideo: true,
          showSkeleton: true,
          showPoints: true,
          showBoundingBox: false,
        },
        net: null,
      }
    };
  },

  mounted() {
    this.init();
  },

  methods: {
    init: async function() {
      const net = await posenet.load();
      const v = this.$refs.video;
      console.log("model loaded");
      this.guiState.net = net;
      this.setupCamera()

      this.detectPoseInRealTime(v, net)
    },

    detectPoseInRealTime: function(video, net) {
      const that = this;
      const videoWidth = 500;
      const videoHeight = videoWidth; 
      const canvas = this.$refs.output;
      const guiState = this.guiState;
      const ctx = canvas.getContext("2d");
      // since images are being fed from a webcam
      const flipHorizontal = true;

      canvas.width = videoWidth;
      canvas.height = videoHeight;

      async function poseDetectionFrame() {
        if (guiState.changeToArchitecture) {
          // Important to purge variables and free up GPU memory
          guiState.net.dispose();

          // Load the PoseNet model weights for either the 0.50, 0.75, 1.00, or 1.01
          // version
          guiState.net = await posenet.load(+guiState.changeToArchitecture);

          guiState.changeToArchitecture = null;
        }

        // Scale an image down to a certain factor. Too large of an image will slow
        // down the GPU
        const imageScaleFactor = guiState.input.imageScaleFactor;
        const outputStride = +guiState.input.outputStride;

        let poses = [];
        let minPoseConfidence;
        let minPartConfidence;
        switch (guiState.algorithm) {
          case "single-pose":
            const pose = await guiState.net.estimateSinglePose(
              video,
              imageScaleFactor,
              flipHorizontal,
              outputStride
            );
            pose.keypoints = pose.keypoints.filter(point => {
              return ["leftShoulder", "leftElbow", "leftWrist"].includes(point.part);
            })
            poses.push(pose);

            minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
            minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
            break;
          case "multi-pose":
            poses = await guiState.net.estimateMultiplePoses(
              video,
              imageScaleFactor,
              flipHorizontal,
              outputStride,
              guiState.multiPoseDetection.maxPoseDetections,
              guiState.multiPoseDetection.minPartConfidence,
              guiState.multiPoseDetection.nmsRadius
            );

            minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
            minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;
            break;
        }

        ctx.clearRect(0, 0, videoWidth, videoHeight);

        if (guiState.output.showVideo) {
          ctx.save();
          ctx.scale(-1, 1);
          ctx.translate(-videoWidth, 0);
          ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
          ctx.restore();
        }

        // For each pose (i.e. person) detected in an image, loop through the poses
        // and draw the resulting skeleton and keypoints if over certain confidence
        // scores
        that.setPoses(poses);
        poses.forEach(({ score, keypoints }) => {
          if (score >= minPoseConfidence) {
            if (guiState.output.showPoints) {
              drawKeypoints(keypoints, minPartConfidence, ctx);
            }
            if (guiState.output.showBoundingBox) {
              drawBoundingBox(keypoints, ctx);
            }
          }
        });
        requestAnimationFrame(poseDetectionFrame);
      }
      poseDetectionFrame();
    },

    setupCamera: async function () {
      const video = this.$refs.video;
      const stream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': {
          facingMode: 'user',
          width: this.videoWidth,
          height: this.videoWidth,
        },
      });
      video.srcObject = stream;
      video.play();
    },

    setPoses(poses){
      this.poses = poses;
      this.addPoseToQueue(poses[0]);
    },

    armStartSnapshot() {
      this.armInitialPosition = this.poses[0];
      console.log('starting pose', this.armInitialPosition);
    },

    addPoseToQueue(pose) {
      if(this.armInitialPosition) {
        this.last50Poses.push(pose);
        if (this.last50Poses.length > 50) {
          this.last50Poses.shift();
        }
      }
      this.checkIfArmGoesDown();
    },

    checkIfArmGoesDown() {
      if( this.last50Poses.length > 10) {
        const previousPosition = this.last50Poses[this.last50Poses.length-2].keypoints[2];
        const currentPosition = this.last50Poses[this.last50Poses.length-1].keypoints[2];
        const probe = previousPosition.score > 0.5 && currentPosition.score > 0.5
        const yAxisTest = previousPosition.position.y < currentPosition.position.y
        if(probe && yAxisTest) {
          this.goDownCounter++
        } else if (!probe || (probe && !yAxisTest)) {
          this.goDownCounter = 0
        }
        if(this.goDownCounter >= 10) {
          console.log('Bet!!!');
          this.goDownCounter = 0;
        } 
      }
    }
    
  }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h3 {
  margin: 40px 0 0;
}
.video {
  display:none;
}
</style>
