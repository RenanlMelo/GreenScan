import React from "react";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { CameraScreen } from "../screens/Scan/Camera/CameraScreen";
import { PreviewScreen } from "../screens/Scan/Preview/PreviewScreen";
import { Report } from "../screens/Report/Report";

export type ScanStackParamList = {
  Camera: undefined;
  Preview: { photoUri: string };
  Report: { reportId: number };
};

const Stack = createNativeStackNavigator<ScanStackParamList>();

export function ScanStack() {
  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      <Stack.Screen name="Camera" component={CameraScreen} />
      <Stack.Screen
        name="Preview"
        options={{ headerShown: false }}
        component={PreviewScreen}
      />
      <Stack.Screen
        name="Report"
        component={Report}
        options={{ headerShown: true, title: "Relatório" }}
      />
    </Stack.Navigator>
  );
}
