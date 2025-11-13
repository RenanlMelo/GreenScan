import React from "react";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { View } from "react-native";
import { HomeStack } from "./Home.routes";
import { ScanStack } from "./Scan.routes";
import { HistoryStack } from "./History.routes";

const Tab = createBottomTabNavigator();

export function BottomTabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarStyle: { height: 120, paddingTop: 12 },
        tabBarShowLabel: false,
        tabBarActiveTintColor: "#517861",
        tabBarInactiveTintColor: "#808080",
        tabBarIcon: ({ focused, color }) => {
          let iconName: keyof typeof MaterialCommunityIcons.glyphMap = "home";
          if (route.name === "Início") iconName = "home";
          else if (route.name === "Scan") iconName = "crop-free";
          else if (route.name === "Histórico") iconName = "history";

          return (
            <View
              style={{
                width: 48,
                height: 48,
                borderRadius: 24,
                backgroundColor: focused ? "#6060604a" : "transparent",
                justifyContent: "center",
                alignItems: "center",
              }}
            >
              <MaterialCommunityIcons name={iconName} size={28} color={color} />
            </View>
          );
        },
      })}
    >
      <Tab.Screen
        name="Início"
        component={HomeStack}
        options={{ headerShown: false }}
      />
      <Tab.Screen
        name="Scan"
        component={ScanStack}
        options={{ headerShown: false }}
      />
      <Tab.Screen
        name="Histórico"
        component={HistoryStack}
        options={{ headerShown: false }}
      />
    </Tab.Navigator>
  );
}
