% see https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/matlab-integration.html

ur5e = loadrobot('universalUR5e');

%*   Set the interface variable and the addresses variable:

% Interface (‘Hardware’, ‘URSim’, ‘Gazebo’)

Interface = "Hardware";
% IP of ROS enabled machine  
ROSDeviceAddress = '192.168.92.132';  
% IP of the robot (if using URSim, the IP address will be 127.0.0.1)  
robotAddress = '192.168.1.10';

username = 'user';

password = 'password';

ROSFolder = '/opt/ros/melodic';

WorkSpaceFolder = '~/ur_ws';

device = rosdevice(ROSDeviceAddress,username,password);

device.ROSFolder = ROSFolder;

generateAndTransferLaunchScriptGettingStarted(device,WorkSpaceFolder,interface,robotAddress);

if ~isCoreRunning(device)  
w = strsplit(system(device, 'who'));  
displayNum = cell2mat(w(2));  
system(device, ['export SVGA\_VGPU10=0;' 'export DISPLAY=' displayNum '.0; ' './launchUR.sh &' ]);  
pause(10);  
end

ur = urROSNode(ROSDeviceAddress,'RigidBodyTree',ur5e)

jointAngles = getJointConfiguration(ur,10)

show(ur.RigidBodyTree,jointAngles);

jointWaypoints = [0 -90 0 -90 0 0]*pi/180;

[result,state] = sendJointConfigurationAndWait(ur,jointWaypoints,'EndTime',5)