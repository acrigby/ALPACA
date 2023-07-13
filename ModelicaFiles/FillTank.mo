within ;
model FillTank
  Modelica.Fluid.Sources.Boundary_pT Source(
    p=3000000,
    T=298.15,
    redeclare package Medium = Modelica.Media.Water.StandardWater,
    nPorts=1)
    annotation (Placement(transformation(extent={{-102,-10},{-82,10}})));
  inner NHES.Fluid.System_TP system(energyDynamics=Modelica.Fluid.Types.Dynamics.SteadyStateInitial)
    annotation (Placement(transformation(extent={{80,78},{100,98}})));
  Modelica.Fluid.Sources.Boundary_pT Sink1(
    p=100000,
    T=298.15,
    redeclare package Medium = Modelica.Media.Water.StandardWater,
    nPorts=1)
    annotation (Placement(transformation(extent={{118,-12},{94,12}})));
  TRANSFORM.Fluid.Volumes.SimpleVolume volume(
    redeclare package Medium = Modelica.Media.Water.StandardWater,
    p_start=1500000,
    T_start=298.15,
    redeclare model Geometry =
        TRANSFORM.Fluid.ClosureRelations.Geometry.Models.LumpedVolume.GenericVolume
        (V=1000),
    use_HeatPort=false)
    annotation (Placement(transformation(extent={{-2,-22},{42,22}})));
  Modelica.Blocks.Sources.RealExpression dLevel(y=volume.medium.p)
    "Heat loss/gain not accounted for in connections (e.g., energy vented to atmosphere) [W]"
    annotation (Placement(transformation(extent={{-122,42},{-76,70}})));
  TRANSFORM.Controls.LimPID PID2(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=5,
    Ti=10000,
    yMax=1.0,
    yMin=0,
    y_start=0.0)
    annotation (Placement(transformation(extent={{-56,82},{-48,90}})));
  TRANSFORM.Fluid.Valves.ValveIncompressible valveIncompressible(
    redeclare package Medium = Modelica.Media.Water.StandardWater,
    dp_nominal=50000,
    m_flow_nominal=50000,
    opening_nominal=1)
    annotation (Placement(transformation(extent={{-46,-12},{-26,8}})));
  TRANSFORM.Fluid.Valves.ValveLinear valveLinear(
    redeclare package Medium = Modelica.Media.Water.StandardWater,
    dp_nominal=100000,
    m_flow_nominal=5000)
    annotation (Placement(transformation(extent={{54,-10},{74,10}})));
  Modelica.Blocks.Sources.Step step1(
    height=0.2,
    offset=0.5,
    startTime=0.5)
    annotation (Placement(transformation(extent={{24,36},{44,56}})));
  Modelica.Blocks.Sources.Constant const(k=1500000)
    annotation (Placement(transformation(extent={{-128,76},{-108,96}})));
equation
  connect(dLevel.y,PID2. u_m) annotation (Line(points={{-73.7,56},{-52,56},{-52,
          81.2}},        color={0,0,127}));
  connect(Source.ports[1], valveIncompressible.port_a) annotation (Line(points=
          {{-82,0},{-64,0},{-64,-2},{-46,-2}}, color={0,127,255}));
  connect(valveIncompressible.port_b, volume.port_a) annotation (Line(points={{
          -26,-2},{-10,-2},{-10,0},{6.8,0}}, color={0,127,255}));
  connect(PID2.y, valveIncompressible.opening)
    annotation (Line(points={{-47.6,86},{-36,86},{-36,6}}, color={0,0,127}));
  connect(volume.port_b, valveLinear.port_a)
    annotation (Line(points={{33.2,0},{54,0}}, color={0,127,255}));
  connect(valveLinear.port_b, Sink1.ports[1])
    annotation (Line(points={{74,0},{94,0}}, color={0,127,255}));
  connect(step1.y, valveLinear.opening)
    annotation (Line(points={{45,46},{64,46},{64,8}}, color={0,0,127}));
  connect(const.y, PID2.u_s)
    annotation (Line(points={{-107,86},{-56.8,86}}, color={0,0,127}));
  annotation (uses(
      NHES(version="2"),
      Modelica(version="4.0.0"),
      TRANSFORM(version="0.5")), experiment(__Dymola_NumberOfIntervals=500000,
        __Dymola_Algorithm="Dassl"));
end FillTank;
