package com.aol.advertising.ml.MLUtil

import org.apache.spark.AccumulatorParam

import scala.collection.mutable.{HashSet => MutableHashSet}

/**
 * Project          : aol-ml
 * File             : Accumulators
 * Author           : Nikhil J Joshi (nikhil.joshi@teamaol.com)
 * Date Created     : 10/26/15
 *
 * Descriptor       : 
 * " The class usage goes here. "
 *
 */

class MutableHashSetAccParam[T]
  extends AccumulatorParam[MutableHashSet[T]] {
  override def zero(initialVal: MutableHashSet[T]): MutableHashSet[T] = MutableHashSet[T]()
  override def addInPlace(s1: MutableHashSet[T],
                          s2: MutableHashSet[T]): MutableHashSet[T] =
    s1 ++ s2
}
