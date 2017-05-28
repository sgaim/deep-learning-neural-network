
var $ = require("jquery");
var _ = require("underscore")
const math = require("mathjs");

class Network{
  constructor() {
    this.layers = [];
    this.names = [];
    this.history_acc = [];
    this.history_loss = [];
  }

  add(current_layer){
    this.layers.push(current_layer);
    this.names.push(current_layer.name+'_'+count.toString());
    count+=1;
    input_dim = current_layer.shape[0];
  }

  compile(do_nothing){
    'Future Works';
  }
}


// Calculate LogSoftmax for Vanilla RNN
class LogSoftmax {
  constructor() {
    this.name = 'LogSoftmax';
  }

  forward(y, gt) {
    var temp = expo(y);
    var prob = temp[gt]/sum(temp);
    var loss = Math.log(prob);
    return -1 * loss;
  }

  backward(y, gt) {
    y = expo(y);
    y = divide(y,sum(y));
    y[gt] = y[gt] - 1;
    return y;
  }
}


// Fully Connected Layer
class FCCell {
  constructor(shape, weightInit = [0,0.1], biasInit = 0, name = 'FCCell') {
    this.shape = shape;
    this.name = name;
    this.W = defaultMatrix(this.shape[0],this.shape[1],weightInit[0],weightInit[1]);
    this.b = Array(this.shape[0]).fill(0);
  }


  forward(x) {
    var first = multiplyMatrices(this.W, x);
    var Y = sum(first,this.b);
    return Y
  }

  backward(x, dv_y) {
    var dv_x = multiplyMatrices(transpose(this.W), dv_y)
    var dv_W = multiplyMatrices(dv_y, transpose(x))
    var dv_b = dv_y

    return [dv_x, dv_W, dv_b]
  }
}


//Vanilla RNN Layer
class RNNCell {
  constructor(shape, weightInit = [0,0.1], biasInit = 0, name = 'RNNCell') {
    this.shape = shape;
    this.name = name;
    this.W = math.random(this.shape,weightInit[0],weightInit[1]);
    this.U = math.random([this.shape[0],this.shape[0]],weightInit[0],weightInit[1]);
    var _form = [];
    for (var i = 0; i < this.shape[0]; i++) {
    	_form.push([0]);
    }
    this.b = math.matrix(_form)._data;
    this.initial_run = true;
  }

  
  forward(x, hprev) {
  	var first = math.multiply(this.W,x._data);
  	if (this.initial_run){
  		var second = math.multiply(this.U, hprev._data);
  		this.initial_run = false;
  	}
  	else{
  		var second = math.multiply(this.U, hprev);	
  	}
  	
  	var z_T = math.add(first,math.matrix(second).reshape([first.length]))._data;
  	var z_t = math.add(z_T,math.matrix(this.b).reshape([first.length]))._data;
    var h_t = math.tanh(z_t);
    return h_t
  }


  
  backward(x, h, hprev, dv_h) {
  	var h_squared = math.square(h);
  	var dv_z = math.multiply(dv_h, math.subtract(1,h_squared));
  	var dv_x = math.dot(math.transpose(this.W), dv_z);
  	var dv_hprev = math.dot(math.transpose(this.U), dv_z);
  	var dv_W = np.dot(dv_z, math.transpose(x));
  	var dv_U = np.dot(dv_z, math.transpose(hprev));
  	var dv_b = dv_z;

    return [dv_x, dv_hprev, dv_W, dv_U, dv_b]
  }
}

function Dense(out, input_shape= input_dim){
	//Will sanitize inputs here.
	return new FCCell([out,input_shape],[]); 
}

function Vanilla(out, input_shape= input_dim){
	//Will sanitize inputs here.
	return new RNNCell([out,input_shape]);
}


// Matrix Operations
// Sum (array)
function sum(arr){
	return arr.reduce(function(a, b) {
		return a + b; }, 0);
}

// Divide  (array, int)
function divide(arr, divisor){
	return arr.map(function(x) {
		return x / divisor; 
	});
}

//Multiplication 1D x int (array, int)
function mult(arr, multiplier){
	return arr.map(function(x) {
		return x * multiplier; 
	});
}

// Log  (array, int)
function log(arr){
	return arr.map(function(x) {
		return Math.log(x); 
	});
}

// Exponenet  //Handles 1D and 2D arrays (array)
function expo(arr) {
	if($.isArray(arr[0])){
		return arr.map(function (x) {
			return x.map(function (y) {
				return Math.exp(y);
			})
		});
	}
	else{
		return arr.map(function (x) {
			return Math.exp(x);
		})
	}
}

// Transpose
function transpose(mat){
  return _.zip.apply(_, mat)
}


//Flatten matrix
function Flatten(mat){
  return [].concat.apply([], mat);
}

// Matrix Multiplication
function multiplyMatrices(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || !a.length || !b.length) {
        throw new Error('arguments should be in 2-dimensional array format');
    }

    var x = a.length,
        z = a[0].length,
        y = b[0].length;

    if (b.length !== z) {
        // XxZ & ZxY => XxY
        throw new Error('number of columns in the first matrix should be the same as the number of rows in the second');
    }

    var productRow = Array.apply(null, new Array(y)).map(Number.prototype.valueOf, 0);
    var product = new Array(x);
    for (var p = 0; p < x; p++) {
        product[p] = productRow.slice();
    }

    for (var i = 0; i < x; i++) {
        for (var j = 0; j < y; j++) {
            for (var k = 0; k < z; k++) {
                product[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return product;
}

//Creating a new Matrix
function defaultMatrix(length, width, min = 0, max = 0.1) {  //min = 
    var matrix = [];
    for (var i=0; i < length; i++) { 
        matrix.push(divide(_.times(width, _.random.bind(min*100-1, max*100+1)),100));
    }
    return matrix;
}

