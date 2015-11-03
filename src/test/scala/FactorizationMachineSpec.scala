import org.apache.spark.ml.classification.FMWeights
import org.apache.spark.mllib.linalg.{DenseMatrix, Vectors}

/**
 * Project          : aol-ml
 * File             : FactorizationMachineSpec
 * Author           : Nikhil J Joshi (nikhil.joshi@teamaol.com)
 * Date Created     : 10/28/15
 *
 * Descriptor       : 
 * " The class usage goes here. "
 *
 */


// Tests for FMweights

//val w1 = new FMWeights(1.0,
//  Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0)),
//  DenseMatrix.eye(3)
//)
//
//val w2 = new FMWeights(1.0,
//  Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0)),
//  DenseMatrix.eye(3)
//)
//
//
//val w3 = w1 + w2 * 0.5
//val w4 = w1 / -0.0 + w2
