import React from "react";
import { Image, Text, View, FlatList, ScrollView } from "react-native";
import { styles } from "./Styles";

type HistoryItem = {
  id: string;
  name: string;
  situation: string;
  time: string;
  image?: string; // optional image url
};

const historyData: HistoryItem[] = [
  { id: "1", name: "Plant A", situation: "Healthy", time: "2h ago" },
  { id: "2", name: "Plant B", situation: "Infected", time: "5h ago" },
  { id: "3", name: "Plant C", situation: "At Risk", time: "1d ago" },
  { id: "4", name: "Plant D", situation: "Healthy", time: "2d ago" },
  { id: "5", name: "Plant E", situation: "Infected", time: "3d ago" },
];

export function RecentHistory() {
  const renderItem = ({ item }: { item: HistoryItem }) => (
    <View style={styles.item}>
      <Image
        style={styles.image}
        source={{
          uri: item.image ?? "https://via.placeholder.com/50", // fallback image
        }}
      />
      <View style={styles.info}>
        <Text style={styles.name}>{item.name}</Text>
        <Text style={styles.situation}>{item.situation}</Text>
      </View>
      <Text style={styles.time}>{item.time}</Text>
    </View>
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Recent History</Text>
      <FlatList
        data={historyData}
        keyExtractor={(item) => item.id}
        renderItem={renderItem}
        showsVerticalScrollIndicator={false}
        scrollEnabled={false}
      />
    </View>
  );
}
