within ;
model DoublePendulum
  "Simple double pendulum with two revolute joints and two bodies"

  parameter Real wstart1=0;
  parameter Real phistart1=-1.57075;
  parameter Real wstart2=0;
  parameter Real phistart2=0;

  extends Modelica.Icons.Example;
  inner Modelica.Mechanics.MultiBody.World world(
    label1="x",
    label2="y",
    n(displayUnit="1") = {0,-1,0})               annotation (Placement(
        transformation(extent={{-126,-10},{-106,10}})));
  Modelica.Mechanics.MultiBody.Joints.Revolute revolute1(
    useAxisFlange=false,
    animation=true,
    phi(fixed=true, start=phistart1),
    w(fixed=true, start=wstart1))
                     annotation (Placement(transformation(extent={{-74,-10},{
            -54,10}})));
  Modelica.Mechanics.MultiBody.Parts.BodyBox boxBody1(r={0.5,0,0}, width=0.06)
    annotation (Placement(transformation(extent={{-20,-10},{0,10}})));
  Modelica.Mechanics.MultiBody.Joints.Revolute revolute2(
    useAxisFlange=true,                                  phi(fixed=true, start=
          phistart2), w(fixed=true, start=wstart2))
                     annotation (Placement(transformation(extent={{20,-10},{40,10}})));
  Modelica.Mechanics.MultiBody.Parts.BodyBox boxBody2(r={0.5,0,0}, width=0.06)
    annotation (Placement(transformation(extent={{60,-10},{80,10}})));
  Modelica.Mechanics.Rotational.Sources.Torque2 torque2
    annotation (Placement(transformation(extent={{-70,50},{-50,70}})));
  Modelica.Blocks.Sources.Constant const(k=1)
    annotation (Placement(transformation(extent={{-170,68},{-150,88}})));
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
  connect(const.y, torque2.tau)
    annotation (Line(points={{-149,78},{-60,78},{-60,64}}, color={0,0,127}));
  connect(torque2.flange_b, revolute2.axis) annotation (Line(points={{-50,60},{
          -10,60},{-10,14},{30,14},{30,10}}, color={0,0,0}));
  connect(torque2.flange_a, revolute2.support) annotation (Line(points={{-70,60},
          {-74,60},{-74,46},{24,46},{24,10}}, color={0,0,0}));
  annotation (
    experiment(
      StopTime=0.2,
      Interval=0.001,
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
    uses(Modelica(version="4.0.0")));
end DoublePendulum;
