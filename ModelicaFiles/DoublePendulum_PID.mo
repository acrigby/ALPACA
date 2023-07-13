within ;
model DoublePendulum_PID
  "Simple double pendulum with two revolute joints and two bodies"

  extends Modelica.Icons.Example;
  inner Modelica.Mechanics.MultiBody.World world(
    label1="x",
    label2="y",
    n(displayUnit="1") = {0,-1,0})               annotation (Placement(
        transformation(extent={{-126,-10},{-106,10}})));
  Modelica.Mechanics.MultiBody.Joints.Revolute revolute1(
    useAxisFlange=true,
    animation=true,
    phi(fixed=true, start=-1.221730476396),
      w(fixed=true)) annotation (Placement(transformation(extent={{-74,-10},{
            -54,10}})));
  Modelica.Mechanics.MultiBody.Parts.BodyBox boxBody1(r={0.5,0,0}, width=0.06)
    annotation (Placement(transformation(extent={{-20,-10},{0,10}})));
  Modelica.Mechanics.MultiBody.Joints.Revolute revolute2(
    useAxisFlange=true,
    phi(fixed=true, start=0),                                             w(
        fixed=true)) annotation (Placement(transformation(extent={{20,-10},{40,10}})));
  Modelica.Mechanics.MultiBody.Parts.BodyBox boxBody2(r={0.5,0,0}, width=0.06)
    annotation (Placement(transformation(extent={{60,-10},{80,10}})));
  Modelica.Mechanics.Rotational.Sources.Torque2 torque2
    annotation (Placement(transformation(extent={{-70,50},{-50,70}})));
  TRANSFORM.Controls.LimPID PID2(
    controllerType=Modelica.Blocks.Types.SimpleController.PD,
    k=5000,
    Ti=500000,
    Td=0.00001,
    y_start=0.0)
    annotation (Placement(transformation(extent={{-130,76},{-122,84}})));
  Modelica.Blocks.Sources.RealExpression dLevel(y=revolute1.phi)
    "Heat loss/gain not accounted for in connections (e.g., energy vented to atmosphere) [W]"
    annotation (Placement(transformation(extent={{-152,60},{-140,72}})));
  Modelica.Blocks.Sources.Constant const(k=-3.1415/2)
    annotation (Placement(transformation(extent={{-196,68},{-176,88}})));
  Modelica.Mechanics.Rotational.Sources.Torque2 torque1
    annotation (Placement(transformation(extent={{20,36},{40,56}})));
  TRANSFORM.Controls.LimPID PID1(
    controllerType=Modelica.Blocks.Types.SimpleController.P,
    k=100,
    Ti=50000,
    Td=0.001,
    y_start=0.0)
    annotation (Placement(transformation(extent={{42,72},{50,80}})));
  Modelica.Blocks.Sources.RealExpression dLevel1(y=revolute2.phi)
    "Heat loss/gain not accounted for in connections (e.g., energy vented to atmosphere) [W]"
    annotation (Placement(transformation(extent={{20,56},{32,68}})));
  Modelica.Blocks.Sources.Constant const1(k=0)
    annotation (Placement(transformation(extent={{-24,64},{-4,84}})));
equation

  connect(revolute2.frame_b, boxBody2.frame_a)
    annotation (Line(
      points={{40,0},{60,0}},
      color={95,95,95},
      thickness=0.5));
  connect(boxBody1.frame_b, revolute2.frame_a)
    annotation (Line(
      points={{0,0},{20,0}},
      color={95,95,95},
      thickness=0.5));
  connect(world.frame_b, revolute1.frame_a) annotation (Line(
      points={{-106,0},{-74,0}},
      color={95,95,95},
      thickness=0.5));
  connect(revolute1.frame_b, boxBody1.frame_a) annotation (Line(
      points={{-54,0},{-20,0}},
      color={95,95,95},
      thickness=0.5));
  connect(torque2.flange_a, revolute1.support) annotation (Line(points={{-70,60},
          {-74,60},{-74,10},{-70,10}}, color={0,0,0}));
  connect(torque2.flange_b, revolute1.axis) annotation (Line(points={{-50,60},{
          -46,60},{-46,16},{-64,16},{-64,10}}, color={0,0,0}));
  connect(dLevel.y, PID2.u_m) annotation (Line(points={{-139.4,66},{-126,66},{
          -126,75.2}}, color={0,0,127}));
  connect(const.y, PID2.u_s) annotation (Line(points={{-175,78},{-152,78},{-152,
          80},{-130.8,80}}, color={0,0,127}));
  connect(PID2.y, torque2.tau)
    annotation (Line(points={{-121.6,80},{-60,80},{-60,64}}, color={0,0,127}));
  connect(dLevel1.y, PID1.u_m)
    annotation (Line(points={{32.6,62},{46,62},{46,71.2}}, color={0,0,127}));
  connect(const1.y, PID1.u_s)
    annotation (Line(points={{-3,74},{-3,76},{41.2,76}}, color={0,0,127}));
  connect(PID1.y, torque1.tau) annotation (Line(points={{50.4,76},{54,76},{54,
          50},{30,50}}, color={0,0,127}));
  connect(torque1.flange_a, revolute2.support)
    annotation (Line(points={{20,46},{16,46},{16,10},{24,10}}, color={0,0,0}));
  connect(torque1.flange_b, revolute2.axis)
    annotation (Line(points={{40,46},{44,46},{44,10},{30,10}}, color={0,0,0}));
  annotation (
    experiment(
      StopTime=10,
      Interval=0.0001,
      __Dymola_Algorithm="Esdirk45a"),
    Documentation(info="<html>
<p>
This example demonstrates that by using joint and body
elements animation is automatically available. Also the revolute
joints are animated. Note, that animation of every component
can be switched of by setting the first parameter <strong>animation</strong>
to <strong>false</strong> or by setting <strong>enableAnimation</strong> in the <strong>world</strong>
object to <strong>false</strong> to switch off animation of all components.
</p>

<blockquote>
<img src=\"modelica://Modelica/Resources/Images/Mechanics/MultiBody/Examples/Elementary/DoublePendulum.png\"
alt=\"model Examples.Elementary.DoublePendulum\">
</blockquote>
</html>"),
    uses(Modelica(version="4.0.0"), TRANSFORM(version="0.5")));
end DoublePendulum_PID;
