import React from "react";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { CameraScreen } from "./Stacks/Camera/CameraScreen";
import { PreviewScreen } from "./Stacks/Preview/PreviewScreen";
import { Report } from "./Stacks/Report/Report";

type RootStackParamList = {
  Camera: undefined;
  Preview: { photoUri: string };
  Report: {
    photoUri: string;
    classe: string;
    confianca: number;
    tratamento: string;
  };
};
const Stack = createNativeStackNavigator<RootStackParamList>();

export function ScanStack() {
  return (
    <Stack.Navigator
      screenOptions={{
        headerShown: false,
        contentStyle: { backgroundColor: "#edf2e1" },
      }}
    >
      <Stack.Screen name="Camera" component={CameraScreen} />
      <Stack.Screen name="Preview" component={PreviewScreen} />
      <Stack.Screen
        name="Report"
        component={Report}
        options={{
          headerShown: true,
          title: "Relatório",
        }}
      />
    </Stack.Navigator>
  );
}
