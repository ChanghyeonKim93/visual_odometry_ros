#ifndef _CONNECTIVITY_NODE_H_
#define _CONNECTIVITY_NODE_H_

#include <iostream>
#include <vector>
#include <memory>

class GraphNodePose;
class GraphNodePoint;
class GraphNodeConstraint;

class BAConnectivityGraph;

typedef std::vector<GraphNodePose>       GraphNodePoseVec;
typedef std::vector<GraphNodePoint>      GraphNodePointVec;
typedef std::vector<GraphNodeConstraint> GraphNodeConstraintVec;

typedef std::shared_ptr<BAConnectivityGraph> BAConnectivityGraphPtr;

/// @brief Point to  poses / constraints 
class GraphNodePose
{
private:
    int index_; // this node index
    std::vector<int> index_of_connected_points_;
    std::vector<int> index_of_connected_constraints_;

    bool flag_connected_to_point_;
    bool flag_connected_to_constraint_;

public:
    GraphNodePose(int index, bool flag_point_connect = true, bool flag_const_connect = false)
    : index_(index), flag_connected_to_point_(flag_point_connect), flag_connected_to_constraint_(flag_const_connect)
    {
        index_of_connected_points_.reserve(2000);
        index_of_connected_constraints_.reserve(10);
    };

    int getNumOfConnectedPoints() const      { return index_of_connected_points_.size(); };
    int getNumOfConnectedConstraints() const { return index_of_connected_constraints_.size(); };


public:
    inline bool isConnectedToPoint() const      { return flag_connected_to_point_; };
    inline bool isConnectedToConstraint() const { return flag_connected_to_constraint_; };
};


class GraphNodePoint
{
private:
    int index_; // this node index
    std::vector<int> index_of_connected_poses_;
    std::vector<int> index_of_connected_constraints_;

    bool flag_connected_to_pose_;
    bool flag_connected_to_constraint_;

public:
    GraphNodePoint(int index, bool flag_pose_connect = true, bool flag_const_connect = false)
    : index_(index), flag_connected_to_pose_(flag_pose_connect), flag_connected_to_constraint_(flag_const_connect)
    {
        index_of_connected_poses_.reserve(2000);
        index_of_connected_constraints_.reserve(10);
    };

    int getNumOfConnectedPoses() const       { return index_of_connected_poses_.size(); };
    int getNumOfConnectedConstraints() const { return index_of_connected_constraints_.size(); };


public:
    inline bool isConnectedToPose() const       { return flag_connected_to_pose_; };
    inline bool isConnectedToConstraint() const { return flag_connected_to_constraint_; };
};


class GraphNodeConstraint
{
private:
    int index_; // this node index
    std::vector<int> index_of_connected_poses_;
    std::vector<int> index_of_connected_points_;
    
    bool flag_connected_to_pose_;
    bool flag_connected_to_point_;

public:
    GraphNodeConstraint(int index, bool flag_pose_connect = true, bool flag_point_connect = false)
    : index_(index), flag_connected_to_pose_(flag_pose_connect), flag_connected_to_point_(flag_point_connect)
    {
        index_of_connected_poses_.reserve(20);
        index_of_connected_points_.reserve(20);
    };

    int getNumOfConnectedPoses() const  { return index_of_connected_poses_.size(); };
    int getNumOfConnectedPoints() const { return index_of_connected_points_.size(); };

public:
    inline bool isConnectedToPose() const  { return flag_connected_to_pose_; };
    inline bool isConnectedToPoint() const { return flag_connected_to_point_; };

};


class BAConnectivityGraph
{
private:
    GraphNodePoseVec       nodes_pose_;
    GraphNodePointVec      nodes_point_;
    GraphNodeConstraintVec nodes_constraint_;

public:
    BAConnectivityGraph()
    {
        nodes_pose_.reserve(500);
        nodes_point_.reserve(50000);
        nodes_constraint_.reserve(500);
    };
    
    ~BAConnectivityGraph() {};

// Get methods
public:
    inline int getNumOfPoses() const       { return nodes_pose_.size(); };
    inline int getNumOfPoints() const      { return nodes_point_.size(); };
    inline int getNumOfConstraints() const { return nodes_constraint_.size(); };


// Set methods
public:
    void setPoseList();
    void setPointList();
    void setConstraintList();

};



#endif