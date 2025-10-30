import React from "react";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import History from "../screens/History/History";
import { Report } from "../screens/Report/Report";

export type HistoryStackParamList = {
  History: undefined;
  Report: { reportId: number };
};

const Stack = createNativeStackNavigator<HistoryStackParamList>();

export function HistoryStack() {
  return (
    <Stack.Navigator screenOptions={{ headerShown: true }}>
      <Stack.Screen name="History" component={History} />
      <Stack.Screen
        name="Report"
        component={Report}
        options={{ headerShown: true, title: "Relatório" }}
      />
    </Stack.Navigator>
  );
}
